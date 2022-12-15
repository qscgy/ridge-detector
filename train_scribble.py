# Adapted from meng-tang/rloss

import argparse
from math import ceil
import os
import numpy as np
from tqdm import tqdm
from yaml import parse
import matplotlib.pyplot as plt

from data import make_data_loader
from torchvision.utils import make_grid

from losses import pixel_matching_loss
from models.deepboundary import DeepBoundary

from rloss.pytorch.deeplabv3plus.mypath import Path
from rloss.pytorch.deeplabv3plus.dataloaders.custom_transforms import denormalizeimage
from rloss.pytorch.deeplabv3plus.modeling.sync_batchnorm.replicate import patch_replication_callback
from rloss.pytorch.deeplabv3plus.modeling.deeplab import *
from rloss.pytorch.deeplabv3plus.utils.loss import SegmentationLosses
from rloss.pytorch.deeplabv3plus.utils.calculate_weights import calculate_weigths_labels
from rloss.pytorch.deeplabv3plus.utils.lr_scheduler import LR_Scheduler
from rloss.pytorch.deeplabv3plus.utils.saver import Saver
from rloss.pytorch.deeplabv3plus.utils.summaries import TensorboardSummary
from rloss.pytorch.deeplabv3plus.utils.metrics import Evaluator
from rloss.pytorch.deeplabv3plus.DenseCRFLoss import DenseCRFLoss
from rloss.pytorch.deeplabv3plus.NormalizedCutLoss import NormalizedCutLoss

class Trainer(object):
    def __init__(self, args):
        self.args = args

        # Define Saver
        self.saver = Saver(args)
        self.saver.save_experiment_config()
        # Define Tensorboard Summary
        self.summary = TensorboardSummary(self.saver.experiment_dir)
        self.writer = self.summary.create_summary()
        self.total_iters = 0    # global_step for logging; the total number of images passed through
        
        # Define Dataloader
        kwargs = {'num_workers': args.workers, 'pin_memory': True}
        self.train_loader, self.val_loader, self.test_loader, self.nclass = make_data_loader(args, **kwargs)

        # Define network
        model = DeepBoundary(num_classes=self.nclass,
                            backbone=args.backbone,
                            output_stride=args.out_stride,
                            sync_bn=args.sync_bn,
                            freeze_bn=args.freeze_bn,
                            boundary=(self.args.bd_loss>0 or self.args.lt_loss>0),
                            in_channels=self.args.in_chan,
                            rw=args.rw,
                            )

        train_params = [{'params': model.get_1x_lr_params(), 'lr': args.lr},
                        {'params': model.get_10x_lr_params(), 'lr': args.lr * 10}]

        # Define Optimizer
        optimizer = torch.optim.SGD(train_params, momentum=args.momentum,
                                    weight_decay=args.weight_decay, nesterov=args.nesterov)

        # Define Criterion
        # whether to use class balanced weights
        if args.use_balanced_weights:
            classes_weights_path = os.path.join(Path.db_root_dir(args.dataset), args.dataset+'_classes_weights.npy')
            if os.path.isfile(classes_weights_path):
                weight = np.load(classes_weights_path)
            else:
                weight = calculate_weigths_labels(args.dataset, self.train_loader, self.nclass)
            weight = torch.from_numpy(weight.astype(np.float32))
        else:
            weight = None
        self.criterion = SegmentationLosses(weight=weight, cuda=args.cuda).build_loss(mode=args.loss_type)
        self.model, self.optimizer = model, optimizer
        
        self.reg_losses = {}

        if args.densecrfloss > 0:
            densecrflosslayer = DenseCRFLoss(weight=args.densecrfloss, sigma_rgb=args.sigma_rgb_crf, sigma_xy=args.sigma_xy_crf, scale_factor=args.rloss_scale)
            print(densecrflosslayer)
            self.reg_losses['dense_crf_loss'] = densecrflosslayer

        if args.ncloss > 0:
            nclayer = NormalizedCutLoss(weight=args.ncloss, sigma_rgb=args.sigma_rgb_nc, sigma_xy=args.sigma_xy_nc, scale_factor=args.rloss_scale)
            print(nclayer)
            self.reg_losses['norm_cut_loss'] = nclayer
        
        # Cross-entropy loss between boundary regions and pixel prediction
        self.boundary_loss = SegmentationLosses(weight=weight, cuda=args.cuda).build_loss(mode='ce')
        
        # Define Evaluator
        self.evaluator = Evaluator(self.nclass)
        # Define lr scheduler
        self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr,
                                            args.epochs, len(self.train_loader))

        # Using cuda
        if args.cuda:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
            patch_replication_callback(self.model)
            self.model = self.model.cuda()

        # Resuming checkpoint
        self.best_pred = 0.0
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'" .format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            if args.cuda:
                self.model.module.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint['state_dict'])
            if not args.ft:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.best_pred = checkpoint['best_pred']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))

        # Clear start epoch if fine-tuning
        if args.ft:
            args.start_epoch = 0

    def training(self, epoch):
        train_loss = 0.0
        train_celoss = 0.0
        self.model.train()
        tbar = tqdm(self.train_loader)
        num_batch_tr = len(self.train_loader)
        softmax = nn.Softmax(dim=1)
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if self.args.in_chan != 3:
                depth = sample['depth']
                image = torch.cat((image, depth), 1)
            croppings = (target!=254).float()
            target[target==254]=255
            # Pixels labeled 255 are those unlabeled pixels. Padded region are labeled 254.
            # see function RandomScaleCrop in dataloaders/custom_transforms.py for the detail in data preprocessing
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()

            self.scheduler(self.optimizer, i, epoch, self.best_pred)
            self.optimizer.zero_grad()

            if self.args.bd_loss>0 or self.args.lt_loss>0:
                output, mask, params = self.model(image)
                mask = mask*croppings[:,None].cuda()
                mask = mask[:,0].bool()
            else:
                output = self.model(image)

            celoss = self.criterion(output, target)
            loss = celoss
            reg_lossvals = {}

            if self.args.bd_loss>0:
                bloss = (self.boundary_loss(output, mask))*self.args.bd_loss
                # bloss = pixel_matching_loss(mask, target)*self.args.bd_loss
                loss = loss + bloss
                reg_lossvals['boundary_loss'] = bloss.item()

            if self.args.lt_loss>0:
                lt_loss = (params[:,1].abs().mean() + (params[:,3]-2*np.pi).abs().mean())*self.args.lt_loss
                loss = loss + lt_loss
                reg_lossvals['long_thin_loss'] = lt_loss.item()

            probs=None
            if self.args.densecrfloss>0 or self.args.ncloss>0:
                probs = softmax(output)
                denormalized_image = denormalizeimage(sample['image'], mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                for ls in self.reg_losses:
                    lv = self.reg_losses[ls](denormalized_image, probs, croppings)
                    if self.args.cuda:
                        lv = lv.cuda()
                    loss = loss + lv
                    reg_lossvals[ls] = lv.item()

            loss.backward()
        
            self.optimizer.step()
            train_loss += loss.item()
            train_celoss += celoss.item()

            self.total_iters += len(image)
            
            tbar.set_description('Train loss: %.3f CE loss %.3f' 
                             % (train_loss / (i + 1),train_celoss / (i + 1)))
            
            self.writer.add_scalar('train/total_loss_iter', loss.item(), self.total_iters)
            self.writer.add_scalar('train/ce_loss', celoss.item(), self.total_iters)
            for l in reg_lossvals:
                self.writer.add_scalar(f'train/{l}', reg_lossvals[l], self.total_iters)

            # Show 10 * 3 inference results each epoch
            if i % (num_batch_tr // 10) == 0:
                self.summary.visualize_image(
                    self.writer, 
                    self.args.dataset, 
                    image[:, :3], target, output, 
                    self.total_iters,
                    regions=mask if self.args.bd_loss>0 else None,
                    )
                if probs is not None:
                    grid = make_grid(probs[:3,1].clone().cpu().unsqueeze(1).data, 3, normalize=False)
                    # self.writer.add_image('Fold_prob', grid, self.total_iters)

        self.writer.add_scalar('train/total_loss_epoch', train_loss, epoch)
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print('Loss: %.3f' % train_loss)

        #if self.args.no_val:
        if self.args.save_interval:
            # save checkpoint every interval epoch
            is_best = False
            if (epoch + 1) % self.args.save_interval == 0:
                self.saver.save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': self.model.module.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'best_pred': self.best_pred,
                }, is_best, filename='checkpoint_epoch_{}.pth.tar'.format(str(epoch+1)))


    def validation(self, epoch):
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.val_loader, desc='\r')
        test_loss = 0.0
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if self.args.in_chan != 3:
                depth = sample['depth']
                image = torch.cat((image, depth), 1)
            target[target==254]=255
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                if self.args.bd_loss>0 or self.args.lt_loss>0:
                    output, mask, params = self.model(image)
                else:
                    output = self.model(image)
            loss = self.criterion(output, target)
            test_loss += loss.item()
            tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))
            pred = output.data.cpu().numpy()
            target = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            # Add batch sample into evaluator
            self.evaluator.add_batch(target, pred)

        # Fast test during the training
        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        self.writer.add_scalar('val/total_loss_epoch', test_loss, epoch)
        self.writer.add_scalar('val/mIoU', mIoU, epoch)
        self.writer.add_scalar('val/Acc', Acc, epoch)
        self.writer.add_scalar('val/Acc_class', Acc_class, epoch)
        self.writer.add_scalar('val/fwIoU', FWIoU, epoch)
        print('Validation:')
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
        print('Loss: %.3f' % test_loss)

        new_pred = mIoU
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)

def main():
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")
    parser.add_argument('--backbone', type=str, default='resnet',
                        choices=['resnet', 'xception', 'drn', 'mobilenet'],
                        help='backbone name (default: resnet)')
    parser.add_argument('--out-stride', type=int, default=16,
                        help='network output stride (default: 8)')
    parser.add_argument('--dataset', type=str, default='pascal',
                        choices=['pascal', 'coco', 'cityscapes'],
                        help='dataset name (default: pascal)')
    parser.add_argument('--use-sbd', action='store_true', default=False,
                        help='whether to use SBD dataset (default: True)')
    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--base-size', type=int, default=216,
                        help='base image size')
    parser.add_argument('--crop-size', type=int, default=216,
                        help='crop image size')
    parser.add_argument('--sync-bn', type=bool, default=None,
                        help='whether to use sync bn (default: auto)')
    parser.add_argument('--freeze-bn', type=bool, default=False,
                        help='whether to freeze bn parameters (default: False)')
    parser.add_argument('--loss-type', type=str, default='ce',
                        choices=['ce', 'focal'],
                        help='loss func type (default: ce)')
    parser.add_argument('--base-dir', type=str, help='base directory for data')
    # training hyper params
    parser.add_argument('--epochs', type=int, default=None, metavar='N',
                        help='number of epochs to train (default: auto)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--batch-size', type=int, default=None,
                        metavar='N', help='input batch size for \
                                training (default: auto)')
    parser.add_argument('--test-batch-size', type=int, default=None,
                        metavar='N', help='input batch size for \
                                testing (default: auto)')
    parser.add_argument('--use-balanced-weights', action='store_true', default=False,
                        help='whether to use balanced weights (default: False)')
    # optimizer params
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (default: auto)')
    parser.add_argument('--lr-scheduler', type=str, default='poly',
                        choices=['poly', 'step', 'cos'],
                        help='lr scheduler mode: (default: poly)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        metavar='M', help='w-decay (default: 5e-4)')
    parser.add_argument('--nesterov', action='store_true', default=False,
                        help='whether use nesterov (default: False)')
    # cuda, seed and logging
    parser.add_argument('--no-cuda', action='store_true', default=
                        False, help='disables CUDA training')
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    # checking point
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--checkname', type=str, default=None,
                        help='set the checkpoint name')
    # finetuning pre-trained models
    parser.add_argument('--ft', action='store_true', default=False,
                        help='finetuning on a different dataset')
    # evaluation option
    parser.add_argument('--eval-interval', type=int, default=1,
                        help='evaluuation interval (default: 1)')
    parser.add_argument('--no-val', action='store_true', default=False,
                        help='skip validation during training')
    # model saving option
    parser.add_argument('--save-interval', type=int, default=None,
                        help='save model interval in epochs')


    # rloss options
    parser.add_argument('--densecrfloss', type=float, default=0,
                        metavar='M', help='densecrf loss (default: 0)')
    parser.add_argument('--ncloss', type=float, default=0,
                        metavar='NC', help='normalized cut loss (default: 0)')
    parser.add_argument('--rloss-scale',type=float,default=1.0,
                        help='scale factor for rloss input, choose small number for efficiency, domain: (0,1]')
    parser.add_argument('--sigma-rgb-crf',type=float,default=15.0,
                        help='DenseCRF sigma_rgb')
    parser.add_argument('--sigma-xy-crf',type=float,default=80.0,
                        help='DenseCRF sigma_xy')
    parser.add_argument('--sigma-rgb-nc', type=float, default=15.0)
    parser.add_argument('--sigma-xy-nc',  type=float, default=40.0)

    parser.add_argument('--bd-loss', type=float, default=0, help='boundary loss weight, or 0 to ignore')
    parser.add_argument('--lt-loss', type=float, default=0, help="'long and thin' loss weight")

    parser.add_argument('--in-chan', type=int, default=3, help='number of input channels')

    parser.add_argument('--rw', action='store_true', default=False, help='use random walks (do not use!)')

    args = parser.parse_args()
    
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    if args.sync_bn is None:
        if args.cuda and len(args.gpu_ids) > 1:
            args.sync_bn = True
        else:
            args.sync_bn = False

    # default settings for epochs, batch_size and lr
    if args.epochs is None:
        epoches = {
            'coco': 30,
            'cityscapes': 200,
            'pascal': 50,
        }
        args.epochs = epoches[args.dataset.lower()]

    if args.batch_size is None:
        args.batch_size = 4 * len(args.gpu_ids)

    if args.test_batch_size is None:
        args.test_batch_size = args.batch_size

    if args.lr is None:
        lrs = {
            'coco': 0.1,
            'cityscapes': 0.01,
            'pascal': 0.007,
        }
        args.lr = lrs[args.dataset.lower()] / (4 * len(args.gpu_ids)) * args.batch_size


    if args.checkname is None:
        args.checkname = 'deeplab-'+str(args.backbone)
    print(args)
    torch.manual_seed(args.seed)
    trainer = Trainer(args)
    print('Starting Epoch:', trainer.args.start_epoch)
    print('Total Epochs:', trainer.args.epochs)
    for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
        trainer.training(epoch)
        if not trainer.args.no_val and epoch % args.eval_interval == (args.eval_interval - 1):
            trainer.validation(epoch)

    trainer.writer.close()

if __name__ == "__main__":
   main()

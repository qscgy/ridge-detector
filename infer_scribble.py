# Adapted from https://github.com/Meng-Tang/rloss

import argparse
import pickle
import os
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from torchvision import transforms
from torchvision.utils import make_grid
from torchvision.datasets import ImageFolder
from torch.autograd import Variable
import torch.nn.functional as F
from os.path import join, isdir
from models.deepboundary import DeepBoundary
import cv2
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

from rloss.pytorch.deeplabv3plus.mypath import Path
from data import FoldSegmentation, MEAN, STDEV, get_depth_from_image
from torch.utils.data import DataLoader, Dataset
from rloss.pytorch.deeplabv3plus.dataloaders.custom_transforms import NormalizeImage, denormalizeimage
from rloss.pytorch.deeplabv3plus.modeling.sync_batchnorm.replicate import patch_replication_callback
from rloss.pytorch.deeplabv3plus.modeling.deeplab import *
from rloss.pytorch.deeplabv3plus.utils.saver import Saver
from rloss.pytorch.deeplabv3plus.utils.loss import SegmentationLosses
import time
import multiprocessing
import glob
from natsort import natsorted
from losses import SwirlLoss

from rloss.pytorch.deeplabv3plus.DenseCRFLoss import DenseCRFLoss
from rloss.pytorch.deeplabv3plus.NormalizedCutLoss import NormalizedCutLoss

global grad_seg

class FixedImageDataset(Dataset):
    """Class representing a dataset of images that does not do any data augmentation.
    Intended for use at inference time.
    """
    def __init__(self, data_path, crop_size, in_channels=3, gt=None, pattern='*.jpg'):
        super().__init__()

        if 'annotations' in data_path:  # annotations file was passed in, which is keyed by file name
            with open(data_path, 'rb') as f:
                self.images = list(pickle.load(f).keys())
        elif data_path[-3:]=='pkl':   # path to a pickle of file names was passed in
            with open(data_path, 'rb') as f:
                self.images = pickle.load(f)
        else:   # data_path is a path to a directory of images
            self.images = glob.glob(os.path.join(data_path, pattern))
        
        self.images = natsorted(self.images)
        self.depths = None
        self.normal_paths = None

        if in_channels == 4:    # using colors and depths
            self.depths = [get_depth_from_image(l) for l in self.images]
        elif in_channels == 6:      # using colors and normals
            with open('normal_paths_test.pkl', 'rb') as f:
                self.normal_paths = pickle.load(f)
        else:   # colors only
            pass

        self.labels = None

        if gt is not None:   # ground truth was provided
            with open(gt, 'rb') as f:
                self.labels = pickle.load(f)
            
        self.crop_size = crop_size
        self.transform = transforms.Compose([
            transforms.Resize((crop_size, crop_size)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STDEV),
        ])

        self.depth_transform = transforms.Compose([
            transforms.Resize((crop_size, crop_size)),
            transforms.ToTensor(),
        ])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        im_name = self.images[index]

        im = Image.open(im_name)
        ims = self.transform(im)

        if self.depths:
            depth = Image.fromarray(np.load(self.depths[index]))
            depth = self.depth_transform(depth)
            ims = torch.cat((ims, depth), 0)
        
        if self.normal_paths:
            normal = np.load(self.normal_paths[im_name])
            normal = transforms.ToTensor()(normal)
            normal = F.interpolate(normal, size=self.crop_size)
            ims = torch.cat((ims, normal), 0)
        
        # Load scribbles
        if self.labels:
            fg_labels, bg_labels = self.labels[im_name]
            labels = np.ones((432, 540, 3))*255
            for l in fg_labels:
                if len(l) > 0:
                    cv2.polylines(labels, np.array([l]), False, (1, 0, 0), 5)
            for l in bg_labels:
                if len(l) > 0:
                    cv2.polylines(labels, np.array([l]), False, (0, 0, 0), 9)
            labels = cv2.resize(labels, (self.crop_size, self.crop_size))
            labels[labels>1] = 255
            labels = labels[:,:,0].astype(np.uint8)
            return ims, labels

        return ims, 0

def process_foldit(results_dir, size):
    foldit_ims = glob.glob(os.path.join(results_dir, '*.png'))
    foldit_ims = natsorted(foldit_ims)
    pred = [f for f in foldit_ims if 'fake_B' in f]
    images = np.zeros((len(pred), *size), dtype=bool)
    for i in range(len(pred)):
        im = Image.open(pred[i])
        im = im.resize(size)
        im = np.array(im)
        images[i] = (im[...,1]==0) & (im[...,2]==0)
    return images

def save_preds(preds, save_dir, images=None, save_np=True):
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    if save_np:
        np.save(os.path.join(save_dir, 'preds.npy'), preds)

    if images is None:
        return
    
    for i in range(preds.shape[0]):
        im = cv2.cvtColor(preds[i].transpose(1,2,0)*255, cv2.COLOR_RGB2BGR)
        path = images[i].split('/')
        cv2.imwrite(os.path.join(save_dir, f'{path[-1][:-4]}.png'), im)

def postprocess(preds):
    output = np.zeros_like(preds)
    for i in range(len(preds)):
        pred = preds[i]
        nblobs, labels, stats, _ = cv2.connectedComponentsWithStats(pred, connectivity=4)

def main(args):
    # Define network
    model = DeepBoundary(num_classes=2,
                            backbone=args.backbone,
                            output_stride=16,
                            sync_bn=None,
                            freeze_bn=None,
                            boundary=(args.bd_loss>0 or args.lt_loss>0),
                            in_channels=args.in_chan,
                            )
    
    # Using cuda
    if not args.no_cuda:
        args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        model = torch.nn.DataParallel(model, device_ids=args.gpu_ids)
        patch_replication_callback(model)
        model = model.cuda()
    
    # load checkpoint
    if not os.path.isfile(args.checkpoint):
        raise RuntimeError("=> no checkpoint found at '{}'" .format(args.checkpoint))
    checkpoint = torch.load(args.checkpoint)
    if args.cuda:
        model.module.load_state_dict(checkpoint['state_dict'], strict=False)
        if 'pam.alpha.k' in checkpoint['state_dict']:
            print(checkpoint['state_dict']['pam.alpha.k'])
    else:
        model.load_state_dict(checkpoint['state_dict'])
    best_pred = checkpoint['best_pred']
    print("=> loaded checkpoint '{}' (epoch {}) best_pred {}"
          .format(args.checkpoint, checkpoint['epoch'], best_pred))
    
    model.eval()

    # Load data
    kwargs = {'num_workers': args.workers, 'pin_memory': True}
    test_data = FixedImageDataset(
        data_path=args.base_dir, 
        crop_size=args.crop_size, 
        in_channels=args.in_chan, 
        gt=args.gt,
        pattern=f'*.{args.extension}'
    )
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, **kwargs)

    # Define input and output arrays
    segmentations = torch.zeros((len(test_loader), args.batch_size, args.crop_size, args.crop_size))
    images = torch.zeros((len(test_loader), args.batch_size, 3, args.crop_size, args.crop_size))

    softmax = nn.Softmax(dim=1)
    labels = torch.zeros_like(segmentations)
    times = []

    for i, (image, label) in enumerate(test_loader):
        if args.gt is not None:     # ground truth labels are provided, so we can calculate metrics
            labels[i] = label
        if not args.no_cuda:
            image = image.cuda()
        start = time.time()
        output = model(image)
        probs = softmax(output)
        elapsed = time.time()-start
        times.append(elapsed)
        print(probs.shape)
        segmentations[i] = probs[:,1].detach().cpu()
        images[i] = denormalizeimage(image[:,:3], MEAN, STDEV)/255.0

    segmentations = segmentations.reshape(segmentations.shape[0]*segmentations.shape[1], *segmentations.shape[2:]).numpy()
    images = images.reshape(images.shape[0]*images.shape[1], *images.shape[2:]).numpy()
    thresh = 0.3    # TODO this is important! Dropping thresh to 0.3 from 0.5 improves accuracy!
    pred = (segmentations > thresh)
    
    foldit_pred_mask = process_foldit(f'/playpen/CEP/results/foldit_public/test_latest/images{"" if not args.sequence else "-test"}', (216, 216))

    # Compute and save metrics if ground-truth labels are provided
    if args.gt is not None:
        labels = labels.reshape(*segmentations.shape)

        # compute metrics
        pred_list = pred[labels<2].astype(np.uint8)
        label_list = labels[labels<2]
        my_report = classification_report(label_list, pred_list, target_names=['Not fold', 'Fold'], output_dict=True)
        foldit_pred_list = foldit_pred_mask[labels<2].astype(np.uint8)
        foldit_report = classification_report(label_list, foldit_pred_list, target_names=['Not fold', 'Fold'], output_dict=True)

        my_accs = np.zeros(segmentations.shape[0])
        for i in range(len(my_accs)):
            pred_i = pred[i][labels[i]<2].astype(np.uint8)
            label_i = labels[i][labels[i]<2]
            report_i = classification_report(label_i, pred_i, target_names=['Not fold', 'Fold'], output_dict=True, zero_division=0)
            my_accs[i] = report_i['accuracy']
            if report_i['Not fold']['recall']==0:
                print('not fold', i, int(report_i['Not fold']['support']*report_i['Not fold']['recall']))
            if report_i['Fold']['recall']==0:
                print('fold', i, int(report_i['Fold']['support']*report_i['Fold']['recall']))
        print(my_accs.mean())
        print(np.std(my_accs))

        fi_accs = np.zeros(segmentations.shape[0])
        for i in range(len(my_accs)):
            pred_i = foldit_pred_mask[i][labels[i]<2].astype(np.uint8)
            label_i = labels[i][labels[i]<2]
            report_i = classification_report(label_i, pred_i, target_names=['Not fold', 'Fold'], output_dict=True)
            fi_accs[i] = report_i['accuracy']
        print(fi_accs.mean())
        print(np.std(fi_accs))

        # Save accuracies for each frame in the same directory as the model weight file
        np.save(os.path.join(*args.checkpoint.split('/')[:-1], 'my_accs.npy'), my_accs)
        np.save(os.path.join(*args.checkpoint.split('/')[:-1], 'fi_accs.npy'), fi_accs)

        my_df = pd.DataFrame(my_report).transpose()
        foldit_df = pd.DataFrame(foldit_report).transpose()
        print('Mine')
        print(my_df)
        print('\nFoldIt')
        print(foldit_df)

        for k1 in ['Not fold', 'Fold']:
            line = ' & '.join([f'(\\textbf{{{my_report[k1][k2]:.2f}}}, {foldit_report[k1][k2]:.2f})' for k2 in ['precision', 'recall', 'f1-score']])
            print(f'{k1} & {line} \\\\ \\hline')
        print(f'Accuracy & & & (\\textbf{{{my_report["accuracy"]:.2f}}}, {foldit_report["accuracy"]:.2f})')
        
    if args.use_examples:
        im_inds = [0, 1, 2, 3, 7, 14]
        if args.sequence:
            im_inds = [0,1,2,3,4]
        ncol = len(im_inds)
    else:
        im_inds = list(range(len(images)))
        ncol = int(np.sqrt(len(im_inds)))

    # Visualize predictions
    _, ax = plt.subplots(1, 3 if args.foldit_path is not None else 2, figsize=(20, 7))
    plt.suptitle(f'Experiment {args.checkpoint.split("/")[-2].split("_")[-1]} with {args.backbone} backbone')
    grid_o = make_grid(torch.from_numpy(images[im_inds]), ncol).numpy().transpose(1,2,0)
    ax[0].imshow(grid_o)
    ax[0].set_title('Input frames')
    ax[0].axis('off')
    
    my_preds = np.copy(images).transpose(1,0,2,3)
    my_preds[0][pred] = 0
    my_preds = my_preds.transpose(1, 0, 2, 3)
    grid_m = make_grid(torch.from_numpy(my_preds[im_inds]), ncol).numpy().transpose(1,2,0)
    ax[1].imshow(grid_m)
    ax[1].set_title('Fold predictions in green')
    ax[1].axis('off')

    if args.foldit_path is not None:
        foldit_preds = np.copy(images).transpose(1,0,2,3)
        foldit_preds[0][foldit_pred_mask] = 0
        foldit_preds = foldit_preds.transpose(1, 0, 2, 3)
        grid_f = make_grid(torch.from_numpy(foldit_preds[im_inds]), ncol).numpy().transpose(1,2,0)
        ax[2].imshow(grid_f)
        ax[2].set_title('Foldit predictions')
        ax[2].axis('off')

    plt.savefig(os.path.join(*(args.checkpoint.split('/')[:-1]), f'eval_thresh_{thresh}.png'), bbox_inches='tight')
    
    # Plot examples with annotations showing positives and negatives
    if args.use_examples:
        foldit_pred_mask_2 = process_foldit(f'/playpen/CEP/results/foldit_internal/test_latest/images{"" if not args.sequence else "-test"}', (216, 216))
        foldit_preds2 = np.copy(images).transpose(1,0,2,3)
        foldit_preds2[0][foldit_pred_mask_2] = 0
        foldit_preds2 = foldit_preds2.transpose(1, 0, 2, 3)

        scribbles = labels[im_inds].squeeze()
        print(np.unique(scribbles))
        im_scribbles = np.copy(images[im_inds]).transpose(0,2,3,1)
        print(scribbles.shape, im_scribbles.shape)
        for i in range(len(im_inds)):
            im_scribbles[i,scribbles[i]==0] = np.array([0,0,1])
            im_scribbles[i,scribbles[i]==1] = np.array([0,1,0])
        im_scribbles = im_scribbles.transpose(0,3,1,2)
        grid_s = make_grid(torch.from_numpy(im_scribbles), ncol).numpy().transpose(1,2,0)

        rois = [((42, 157),(114,210), (0,1,0), 2),
        ((397,8), (426,71), (0,1,0), 2),
        ((493, 98), (547, 175), (0,1,0), 2),
        ((818, 135), (868, 202), (1,0,0), 2),
        ((1246, 91), (1307, 213), (1,0,0), 2),
        ]
        if not args.sequence:
            for r in rois:
                # grid_o = cv2.rectangle(grid_o.copy(), *r)
                grid_m = cv2.rectangle(grid_m.copy(), *r)
                grid_f = cv2.rectangle(grid_f.copy(), *r)
            #     grid_f2 = cv2.rectangle(grid_f2.copy(), *r)
            grid_m = cv2.ellipse(grid_m.copy(), (344, 169), (35, 25), 0, 0, 360, (0.6,0,1), 3)

        plt.figure(figsize=(13,6))
        plt.imshow(np.vstack((grid_o,grid_m,grid_f,grid_s)))
        plt.xticks([])
        plt.yticks([])
        plt.text(-20,216/2, 'Original', fontsize=22, horizontalalignment='right', verticalalignment='center')
        plt.text(-20, 216*1.5, 'Ours', fontsize=22, horizontalalignment='right', verticalalignment='center')
        plt.text(-20, 216*2.5, 'FoldIt', fontsize=22, horizontalalignment='right', verticalalignment='center')
        plt.text(-20, 216*3.5, 'Scribbles', fontsize=22, horizontalalignment='right', verticalalignment='center')

        plt.savefig(os.path.join(*(args.checkpoint.split('/')[:-1]), f'examples_internal_public{"" if not args.sequence else "-seq"}.png'), bbox_inches='tight')

    preds_dir = os.path.join(args.outdir, 'results-mine')
    foldit_dir = os.path.join(args.outdir, 'results-foldit-internal')
    save_preds(segmentations, preds_dir, test_data.images if args.use_examples else None)
    if args.gt is not None:
        if not os.path.isdir(foldit_dir) and args.foldit_path is not None:
            save_preds(foldit_preds, foldit_dir, test_data.images)

    return None

def find_best(args):
    means = np.zeros(96)
    stds = np.zeros(96)
    accs = np.zeros(96)
    chkdir = os.path.join(*args.checkpoint.split('/')[:-1])

    for i in range(1,97):
        args.gpu_ids='0'
        args.checkpoint = os.path.join(chkdir, f'checkpoint_epoch_{i}.pth.tar')
        means[i-1], stds[i-1], accs[i-1] = main(args)
    np.savez(os.path.join(chkdir, 'evals.npz'), means=means, stds=stds, accs=accs)

    plt.errorbar(np.arange(1,97), means, stds, fmt='o')
    plt.title('Average accuracy at epoch on test set')
    plt.xlabel('Epoch')
    plt.ylabel('Mean accuracy')
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Inference")
    parser.add_argument('--backbone', type=str, default='resnet',
                        choices=['resnet', 'xception', 'drn', 'mobilenet'],
                        help='backbone name (default: resnet)')
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--n_class', type=int, default=21)
    parser.add_argument('--crop_size', type=int, default=513,
                        help='crop image size')
    parser.add_argument('--no_cuda', action='store_true', default=
                        False, help='disables CUDA training')
    # checking point
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='put the path to checkpoint if needed')
    # rloss options
    parser.add_argument('--rloss-weight', type=float,
                        metavar='M', help='densecrf loss')
    parser.add_argument('--rloss-scale',type=float,default=0.5,
                        help='scale factor for rloss input, choose small number for efficiency, domain: (0,1]')
    parser.add_argument('--sigma-rgb',type=float,default=15.0,
                        help='DenseCRF sigma_rgb')
    parser.add_argument('--sigma-xy',type=float,default=80.0,
                        help='DenseCRF sigma_xy')
    parser.add_argument('--batch-size', type=int, default=5)
    
    # output directory
    parser.add_argument('--output_directory', type=str,
                        help='output directory')

    # input image directory
    parser.add_argument('--base-dir', type=str, help='image directory')

    parser.add_argument('--bd-loss', type=float, default=0, help='boundary loss weight, or 0 to ignore')
    parser.add_argument('--lt-loss', type=float, default=0, help="'long and thin' loss weight")

    parser.add_argument('--in-chan', type=int, default=3, help='number of input channels')

    parser.add_argument('--sequence', action='store_true', help='use 019 sequence images')
    parser.add_argument('--use-examples', action='store_true', help='use preselected examples or all test images')
    parser.add_argument('--figures', action='store_true', help='generate figures')
    parser.add_argument('--gt', type=str, default=None)
    parser.add_argument('--foldit-path', type=str, default=None)
    parser.add_argument('--outdir', type=str, default='.', help='output directory for images')
    parser.add_argument('--extension', '-e', type=str, default='jpg', help='input image file extension')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    main(args)
    # find_best(args)

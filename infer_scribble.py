# Adapted from https://github.com/Meng-Tang/rloss

import argparse
import pickle
import os
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.utils import make_grid
from torchvision.datasets import ImageFolder
from torch.autograd import Variable
from os.path import join, isdir
from models.deepboundary import DeepBoundary
import cv2

from rloss.pytorch.deeplabv3plus.mypath import Path
from data import FoldSegmentation, MEAN, STDEV, get_depth_from_image
from torch.utils.data import DataLoader, Dataset
from rloss.pytorch.deeplabv3plus.dataloaders.custom_transforms import NormalizeImage, denormalizeimage
from rloss.pytorch.deeplabv3plus.modeling.sync_batchnorm.replicate import patch_replication_callback
from rloss.pytorch.deeplabv3plus.modeling.deeplab import *
from rloss.pytorch.deeplabv3plus.utils.saver import Saver
import time
import multiprocessing
import glob
from natsort import natsorted

from rloss.pytorch.deeplabv3plus.DenseCRFLoss import DenseCRFLoss
from rloss.pytorch.deeplabv3plus.NormalizedCutLoss import NormalizedCutLoss

global grad_seg

class FixedImageDataset(Dataset):
    def __init__(self, crop_size, in_channels=3, gt=False):
        super().__init__()
        # self.images = glob.glob(os.path.join(root, '*.jpg'))[:25]
        # print(self.images)
        # self.images = ['/playpen/Datasets/scribble-test/testA/011_frame014577.jpg', '/playpen/Datasets/scribble-test/testA/061_frame008798.jpg', '/playpen/Datasets/scribble-test/testA/030_frame021700.jpg', '/playpen/Datasets/scribble-test/testA/036_frame028843.jpg', '/playpen/Datasets/scribble-test/testA/Auto_A_Nov01_11-50-02_001_frame037529.jpg', '/playpen/Datasets/scribble-test/testA/015_frame017096.jpg', '/playpen/Datasets/scribble-test/testA/028_frame020968.jpg', '/playpen/Datasets/scribble-test/testA/047_frame013507.jpg', '/playpen/Datasets/scribble-test/testA/022_frame021904.jpg', '/playpen/Datasets/scribble-test/testA/Auto_A_Nov12_13-56-23_001_frame024463.jpg', '/playpen/Datasets/scribble-test/testA/059_frame008342.jpg', '/playpen/Datasets/scribble-test/testA/066_frame016078.jpg', '/playpen/Datasets/scribble-test/testA/008_frame033659.jpg', '/playpen/Datasets/scribble-test/testA/007_frame014436.jpg', '/playpen/Datasets/scribble-test/testA/008_frame034035.jpg', '/playpen/Datasets/scribble-test/testA/060_frame026291.jpg', '/playpen/Datasets/scribble-test/testA/011_frame014067.jpg', '/playpen/Datasets/scribble-test/testA/058_frame159460.jpg', '/playpen/Datasets/scribble-test/testA/Auto_A_Oct18_13-33-20_002_frame033516.jpg', '/playpen/Datasets/scribble-test/testA/Auto_A_Feb08_12-20-44_001_frame007594.jpg', '/playpen/Datasets/scribble-test/testA/039_frame007386.jpg', '/playpen/Datasets/scribble-test/testA/039_frame007252.jpg', '/playpen/Datasets/scribble-test/testA/060_frame026260.jpg', '/playpen/Datasets/scribble-test/testA/005_frame053501.jpg', '/playpen/Datasets/scribble-test/testA/Auto_A_Nov01_11-50-02_001_frame037552.jpg']
        with open('test_set.pkl', 'rb') as f:
            self.images = pickle.load(f)
        self.images = natsorted(self.images)
        self.depths = None
        if in_channels > 3:
            self.depths = [get_depth_from_image(l) for l in self.images]
            # for i in range(len(self.depths)):
            #     fname = self.depths[i]
            #     path = fname.split('/')
            #     rdir = ('/' if fname[0]=='/' else '') + os.path.join(*path[:-3], 'geodepth2')
            #     ddir = '_'.join(path[-1].split('_')[:-2])
            #     self.depths[i] = os.path.join(rdir, ddir, 'colon_norm_preall_abs_nosm', path[-1][len(ddir)+1:])
        
        self.labels = None
        if gt:
            with open('test_set.pkl', 'rb') as f:
                self.labels = pickle.load(f)
            

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
            # print(depth.shape)
            # print(ims.shape)
            return torch.cat((ims, depth), 0)
        
        if self.labels:
            fg_labels, bg_labels = self.labels[im_name]
            labels = np.ones((432, 540, 3))*255
            for l in fg_labels:
                if len(l) > 0:
                    cv2.polylines(labels, np.array([l]), False, (1, 0, 0), self.fstroke)
            for l in bg_labels:
                if len(l) > 0:
                    cv2.polylines(labels, np.array([l]), False, (0, 0, 0), self.bstroke)
        return ims

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

def save_preds(preds, images, save_dir):
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
        for i in range(preds.shape[0]):
            im = cv2.cvtColor(preds[i].transpose(1,2,0)*255, cv2.COLOR_RGB2BGR)
            # cv2.imshow(' ', im)
            # cv2.waitKey(0)
            path = images[i].split('/')
            # print(os.path.join(save_dir, f'{path[-3]}_{path[-1][:-4]}.png'))
            cv2.imwrite(os.path.join(save_dir, f'{path[-3]}_{path[-1][:-4]}.jpg'), im)

def main():

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

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    
    # Define Dataloader
    kwargs = {'num_workers': args.workers, 'pin_memory': True}
    
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
        model.module.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint['state_dict'])
    best_pred = checkpoint['best_pred']
    print("=> loaded checkpoint '{}' (epoch {}) best_pred {}"
          .format(args.checkpoint, checkpoint['epoch'], best_pred))
    
    model.eval()

    kwargs = {'num_workers': args.workers, 'pin_memory': True}
    test_data = FixedImageDataset(crop_size=args.crop_size, in_channels=args.in_chan)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, **kwargs)
    segmentations = torch.zeros((len(test_loader), args.batch_size, args.crop_size, args.crop_size))
    regions = torch.zeros(segmentations.shape)
    images = torch.zeros((len(test_loader), args.batch_size, 3, args.crop_size, args.crop_size))
    softmax = nn.Softmax(dim=1)

    for i, image in enumerate(test_loader):
        if not args.no_cuda:
            image = image.cuda()
        output = model(image)
        probs = softmax(output)
        segmentations[i] = probs[:,1].detach().cpu()
        # images[i] = unnorm.detach().cpu()
        images[i] = denormalizeimage(image[:,:3], MEAN, STDEV)/255.0
        # regions[i] = mask.squeeze().detach().cpu()

    segmentations = segmentations.reshape(segmentations.shape[0]*segmentations.shape[1], *segmentations.shape[2:]).numpy()
    images = images.reshape(images.shape[0]*images.shape[1], *images.shape[2:]).numpy()
    thresh = 0.5
    pred = (segmentations > thresh)

    # visualize prediction
    fig, ax = plt.subplots(1, 3, figsize=(20, 7))
    plt.suptitle(f'Experiment {args.checkpoint.split("/")[-2].split("_")[-1]} with {args.backbone} backbone')
    grid = make_grid(torch.from_numpy(images), int(np.sqrt(images.shape[0]))).numpy().transpose(1,2,0)
    ax[0].imshow(grid)
    ax[0].set_title('Input frames')
    ax[0].axis('off')
    
    my_preds = np.copy(images).transpose(1,0,2,3)
    my_preds[0][pred] = 0
    my_preds = my_preds.transpose(1, 0, 2, 3)
    grid = make_grid(torch.from_numpy(my_preds), int(np.sqrt(my_preds.shape[0]))).numpy().transpose(1,2,0)
    ax[1].imshow(grid)
    ax[1].set_title('Fold predictions in green')
    ax[1].axis('off')

    foldit_pred_mask = process_foldit('/playpen/CEP/results/foldit_public/test_latest/images', (216, 216))
    foldit_preds = np.copy(images).transpose(1,0,2,3)
    foldit_preds[0][foldit_pred_mask] = 0
    foldit_preds = foldit_preds.transpose(1, 0, 2, 3)
    grid = make_grid(torch.from_numpy(foldit_preds), int(np.sqrt(foldit_preds.shape[0]))).numpy().transpose(1,2,0)
    ax[2].imshow(grid)
    ax[2].set_title('Foldit predictions')
    ax[2].axis('off')

    # regions = regions.reshape(-1, 1, args.crop_size, args.crop_size)
    # print(regions.min(), regions.max())
    # grid = make_grid(regions, int(np.sqrt(regions.shape[0])), scale_each=True).numpy().transpose(1,2,0)
    # print(grid.dtype)
    # ax[1][0].imshow(grid)
    # ax[1][0].set_title('Region masks')
    # ax[1][0].axis('off')

    plt.savefig(os.path.join(*(args.checkpoint.split('/')[:-1]), f'eval_thresh_{thresh}.png'), bbox_inches='tight')

    preds_dir = os.path.join(*(args.checkpoint.split('/')[:-1]), 'results-mine')
    foldit_dir = os.path.join(*(args.checkpoint.split('/')[:-1]), 'results-foldit')
    orig_dir = os.path.join(*(args.checkpoint.split('/')[:-1]), 'original')
    save_preds(my_preds, test_data.images, preds_dir)
    save_preds(foldit_preds, test_data.images, foldit_dir)
    save_preds(images, test_data.images, orig_dir)

    # plt.figure(FIGSIZE=(14, 7))
    # plt.title(f'P(fold) for experiment {args.checkpoint.split("/")[-2].split("_")[-1]} with {args.backbone} backbone')
    # grid = segmentations
    # plt.imshow(segmentations[0,1].detach().cpu().numpy(), cmap='jet')
    # plt.colorbar()

    plt.show()
    

if __name__ == "__main__":
   main()

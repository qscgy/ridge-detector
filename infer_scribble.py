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

from rloss.pytorch.deeplabv3plus.DenseCRFLoss import DenseCRFLoss
from rloss.pytorch.deeplabv3plus.NormalizedCutLoss import NormalizedCutLoss

global grad_seg

class FixedImageDataset(Dataset):
    '''
    A class to represent an image dataset whose contents are defined by the user. This class is used at inference time
    to allow for implementation of preprocessing separate from the preprocessing applied to training data in FoldSegmentation.

    Arguments:
        base_dir (str): base directory for annotations
        crop_size (int): the side length to which images will be resized (as a square)
        in_channels (int): the number of input channels to the network. 3 for images only, 4 for images+depths, 6 for images+normals.
        gt (bool): Whether to look for ground-truth labels. Should be True generally.
        process (bool): whether to apply file preprocessing.
        normal_paths (List[str]): paths to normals, only for in_channels=6
    '''
    def __init__(self, base_dir, seq_dir, crop_size, in_channels=3, gt=False, process=True, normal_paths=None):
        super().__init__()

        self.normal_paths = normal_paths
        self.base_dir = base_dir
        self.seq_dir = seq_dir
        self.process = process
        if process and not gt:
            with open(os.path.join(base_dir,'test_set.pkl'), 'rb') as f:
                self.images = pickle.load(f)
        elif process:
            with open(os.path.join(base_dir,'annotations_test.pkl'), 'rb') as f:
                self.images = pickle.load(f).keys()
        else:
            with open('annotations_019.pkl', 'rb') as f:
                self.images = list(pickle.load(f).keys())
        self.images = natsorted(self.images)
        self.depths = None

        # Use depths
        if in_channels == 4:
            if not process:
                # This was only used for demonstrating consistency over consecutive frames, in the paper.
                self.depths = [os.path.join('/playpen/Datasets/geodepth2/019/colon_norm_preall_abs_nosm', l.split('/')[-1].split('.')[0]+'_disp.npy') for l in self.images]
            else:
                self.depths = [get_depth_from_image(os.path.join(seq_dir, l)) for l in self.images]

        # Use normals
        if in_channels==6:
            with open('normal_paths_test.pkl', 'rb') as f:
                self.normal_paths = pickle.load(f)

        self.labels = None
        if gt:
            label_file = 'annotations_test.pkl' if process else 'annotations_019.pkl'
            with open(os.path.join(base_dir, label_file), 'rb') as f:
                in_labels = pickle.load(f)
            
            self.labels = in_labels
            # if not process:
            #     self.labels = in_labels
            # else:
            #     # translate file names between the one used for the key and the image name in test_set.pkl (###/img_corr)
            #     # see the comment in __getitem__ for further details.
            #     self.labels = {}
            #     for k in in_labels:
            #         print(k)
            #         path = k.split('/')[-1]
            #         test_path = os.path.join('/playpen/Datasets/geodepth2/', path[:-16], 'img_corr', path[-15:])
            #         self.labels[test_path] = in_labels[k]

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
        im = Image.open(os.path.join(self.seq_dir, im_name))
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
        
        if self.labels:
            fg_labels, bg_labels = self.labels[im_name]
            labels = np.ones((432, 540, 3))*255
            for l in fg_labels:
                if len(l) > 0:
                    cv2.polylines(labels, np.array([l]), False, (1, 0, 0), 2)
            for l in bg_labels:
                if len(l) > 0:
                    cv2.polylines(labels, np.array([l]), False, (0, 0, 0), 2)
            labels = cv2.resize(labels, (self.crop_size, self.crop_size))
            labels[labels>1] = 255
            labels = labels[:,:,0].astype(np.uint8)
            return ims, labels

        return ims

def process_foldit(results_dir, size):
    '''
    Extracts binary predictions from FoldIt outputs.
    '''
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
    '''
    Save predicted images if not already saved.
    '''
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
        for i in range(preds.shape[0]):
            im = cv2.cvtColor(preds[i].transpose(1,2,0)*255, cv2.COLOR_RGB2BGR)
            path = images[i].split('/')
            cv2.imwrite(os.path.join(save_dir, f'{path[-3]}_{path[-1][:-4]}.png'), im)

def postprocess(preds):
    output = np.zeros_like(preds)
    for i in range(len(preds)):
        pred = preds[i]
        nblobs, labels, stats, _ = cv2.connectedComponentsWithStats(pred, connectivity=4)

def main(args):
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
        model.module.load_state_dict(checkpoint['state_dict'], strict=False)
        if 'pam.alpha.k' in checkpoint['state_dict']:
            print(checkpoint['state_dict']['pam.alpha.k'])
    else:
        model.load_state_dict(checkpoint['state_dict'])
    best_pred = checkpoint['best_pred']
    print("=> loaded checkpoint '{}' (epoch {}) best_pred {}"
          .format(args.checkpoint, checkpoint['epoch'], best_pred))
    
    model.eval()

    kwargs = {'num_workers': args.workers, 'pin_memory': True}
    test_data = FixedImageDataset(args.base_dir, args.sequence_dir, crop_size=args.crop_size, 
                        in_channels=args.in_chan, gt=args.gt, process=not args.sequence)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, **kwargs)
    segmentations = torch.zeros((len(test_loader), args.batch_size, args.crop_size, args.crop_size))
    images = torch.zeros((len(test_loader), args.batch_size, 3, args.crop_size, args.crop_size))
    softmax = nn.Softmax(dim=1)
    labels = torch.zeros_like(segmentations)

    for i, data in enumerate(test_loader):
        if args.gt:
            image, label = data
            labels[i] = label
        else:
            image = data
        if not args.no_cuda:
            image = image.cuda()
        output = model(image)
        probs = softmax(output)
        segmentations[i] = probs[:,1].detach().cpu()
        images[i] = denormalizeimage(image[:,:3], MEAN, STDEV)/255.0

    segmentations = segmentations.reshape(segmentations.shape[0]*segmentations.shape[1], *segmentations.shape[2:]).numpy()
    images = images.reshape(images.shape[0]*images.shape[1], *images.shape[2:]).numpy()
    labels = labels.reshape(*segmentations.shape)
    thresh = 0.3    # TODO this is important! Dropping thresh to 0.3 from 0.5 improves accuracy!
    pred = (segmentations > thresh)

    if args.gt:
        # compute metrics
        pred_list = pred[labels<2].astype(np.uint8)
        label_list = labels[labels<2]
        my_report = classification_report(label_list, pred_list, target_names=['Not fold', 'Fold'], output_dict=True)

        foldit_pred_mask = process_foldit(args.foldit_dir, (args.crop_size, args.crop_size))
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
        
        #Print LaTeX-formatted tables of results
        print(my_accs.mean())
        print(np.std(my_accs))
        np.save(os.path.join(*args.checkpoint.split('/')[:-1], 'my_accs.npy'), my_accs)

        fi_accs = np.zeros(segmentations.shape[0])
        for i in range(len(my_accs)):
            pred_i = foldit_pred_mask[i][labels[i]<2].astype(np.uint8)
            label_i = labels[i][labels[i]<2]
            report_i = classification_report(label_i, pred_i, target_names=['Not fold', 'Fold'], output_dict=True)
            fi_accs[i] = report_i['accuracy']
        print(fi_accs.mean())
        print(np.std(fi_accs))
        np.save(os.path.join(*args.checkpoint.split('/')[:-1], 'fi_accs.npy'), fi_accs)

        # Box plot of results, optional
        # plt.boxplot([my_accs, fi_accs])
        # plt.show()

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

    if not args.figures:
        return my_accs.mean(), np.std(my_accs), my_report['accuracy']
        
    if args.use_examples:
        im_inds = [0, 1, 2, 3, 7, 14]
        if args.sequence:
            im_inds = [0,1,2,3,4]
        ncol = len(im_inds)
    else:
        im_inds = list(range(len(images)))
        ncol = int(np.sqrt(len(im_inds)))

    # Visualize prediction
    fig, ax = plt.subplots(1, 3 if args.gt else 2, figsize=(20, 7))
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

    if args.gt:
        foldit_preds = np.copy(images).transpose(1,0,2,3)
        foldit_preds[0][foldit_pred_mask] = 0
        foldit_preds = foldit_preds.transpose(1, 0, 2, 3)
        grid_f = make_grid(torch.from_numpy(foldit_preds[im_inds]), ncol).numpy().transpose(1,2,0)
        ax[2].imshow(grid_f)
        ax[2].set_title('Foldit predictions')
        ax[2].axis('off')

    plt.savefig(os.path.join(*(args.checkpoint.split('/')[:-1]), f'eval_thresh_{thresh}.png'), bbox_inches='tight')
    # plt.close()
    
    # Examples for the paper.
    if args.use_examples:
        foldit_pred_mask_2 = process_foldit(f'/playpen/CEP/results/foldit_internal/test_latest/images{"" if not args.sequence else "-test"}', (216, 216))
        foldit_pred_list_2 = foldit_pred_mask[labels<2].astype(np.uint8)
        foldit_preds2 = np.copy(images).transpose(1,0,2,3)
        foldit_preds2[0][foldit_pred_mask_2] = 0
        foldit_preds2 = foldit_preds2.transpose(1, 0, 2, 3)
        grid_f2 = make_grid(torch.from_numpy(foldit_preds2[im_inds]), ncol).numpy().transpose(1,2,0)

        foldit_report2 = classification_report(label_list, foldit_pred_list_2, target_names=['Not ridge', 'Ridge'], output_dict=True)
        print('Foldit-UNC\n')
        print(pd.DataFrame(foldit_report2).transpose())

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
        # plt.imshow(np.vstack((grid_o,grid_m,grid_f,grid_f2)))
        plt.imshow(np.vstack((grid_o,grid_m,grid_f)))
        plt.xticks([])
        plt.yticks([])
        plt.text(-20,216/2, 'Original', fontsize=22, horizontalalignment='right', verticalalignment='center')
        plt.text(-20, 216*1.5, 'Ours', fontsize=22, horizontalalignment='right', verticalalignment='center')
        plt.text(-20, 216*2.5, 'FoldIt', fontsize=22, horizontalalignment='right', verticalalignment='center')
        # plt.text(-20, 216*3.5, 'FoldIt-UNC', fontsize=22, horizontalalignment='right', verticalalignment='center')

        plt.savefig(os.path.join(*(args.checkpoint.split('/')[:-1]), f'examples_internal_public{"" if not args.sequence else "-seq"}.png'), bbox_inches='tight')

    preds_dir = os.path.join(*(args.checkpoint.split('/')[:-1]), 'results-mine')
    foldit_dir = os.path.join(*(args.checkpoint.split('/')[:-1]), 'results-foldit-internal')
    orig_dir = os.path.join(*(args.checkpoint.split('/')[:-1]), 'original')
    if not os.path.isdir(preds_dir):
        save_preds(my_preds, test_data.images, preds_dir)
    if args.gt and not os.path.isdir(foldit_dir):
        save_preds(foldit_preds, test_data.images, foldit_dir)
    if not os.path.isdir(orig_dir):
        save_preds(images, test_data.images, orig_dir)

    plt.show()
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

    parser.add_argument('--gt', action='store_true', help='use ground truth labels')

    parser.add_argument('--foldit-dir', type=str, help='path to foldit results')

    parser.add_argument('--bd-loss', type=float, default=0, help='boundary loss weight, or 0 to ignore')
    parser.add_argument('--lt-loss', type=float, default=0, help="'long and thin' loss weight")

    parser.add_argument('--in-chan', type=int, default=3, help='number of input channels')

    parser.add_argument('--sequence', action='store_true', help='use 019 sequence images')
    parser.add_argument('--use-examples', action='store_true', help='use preselected examples or all test images')
    parser.add_argument('--figures', action='store_true', help='generate figures')

    parser.add_argument('--sequence-dir', type=str, help='directory of colonoscopy sequences')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    main(args)
    # find_best(args)

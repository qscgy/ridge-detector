from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from rloss.pytorch.deeplabv3plus.dataloaders.custom_transforms import *
from PIL import Image
import cv2
import numpy as np
import os
import glob
import pickle
from natsort import natsorted
import random

# MEAN = [0.55615947, 0.6482298, 0.84056612]
# STDEV = [0.15693977, 0.16230874, 0.15292975]
MEAN = [0.485, 0.456, 0.406]
STDEV = [0.229, 0.224, 0.225]

def make_data_loader(*args, **kwargs):
    '''
    Construct DataLoaders for training and validation.

    Returns:
        (DataLoader): train DataLoader
        (DataLoader): validation DataLoader
        (NoneType): test DataLoader, not implemented
        (int): number of classes
    '''
    instance = ('instance' in kwargs and kwargs['instance']==True)
    train_set = FoldSegmentation(args, split='train', instance=instance)
    val_set = FoldSegmentation(args, split='val', instance=instance)
    args = args[0]

    num_class = train_set.NUM_CLASSES
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
    test_loader = None

    return train_loader, val_loader, test_loader, num_class

def get_depth_from_image(fname, model_name='colon_norm_preall_abs_nosm'):
    '''
    Find the ColDE depth maps for a given image.
    
    Arguments:
        fname (str): image file name
        modle_name (str): ColDE model name; should be the folder where ColDE outputs depth predictions in the sequence folder.
    Returns:
        (str): file name of depth map
    '''
    path = fname.split('/')
    frame = path[-1].split('.')[0]
    depth = os.path.join(*path[:-2], model_name, f'{frame}_disp.npy')
    if fname[0]=='/':
        depth = '/'+depth
    return depth

def frame_num_to_index(fpath, frame):
    img_dir = os.path.dirname(fpath)
    zero_frame = int(natsorted(os.listdir(img_dir))[0][5:11])
    return frame-zero_frame

def get_normals_from_image(fname):
    path = fname.split('/')
    frame = int(path[-1].split('.')[0][5:])
    index = frame_num_to_index(fname, frame)
    num = path[-3]
    normals = os.path.join(*path[:-2], 'NFPS', 'images', f'{num}_{index:03d}', f'{num}_{index:03d}_nr_pred.npy')
    if fname[0]=='/':
        normals = '/'+normals
    return normals

class FoldSegmentation(Dataset):
    '''
    A class to represent a dataset of labeled colon images for training and validation.

    Arguments:
        _args (argparse.Namespace): command-line args passed to train_scribble.py
        split (str): the dataset splt. Must be 'train' or 'val'.
        fstroke (int): desired stroke width of fold scribbles, in pixels.
        bstroke (int): desired stroke width of not-fold scribbles, in pixels.
        instance (bool): is instance segmenation. Not implemented.
    '''
    NUM_CLASSES = 2
    def __init__(
        self,
        _args,
        split='train',
        fstroke=5,
        bstroke=9,
        instance=False,
    ):
        super().__init__()
        random.seed(1917)
        if isinstance(_args, tuple):
                self.args = _args[0]
        else:
            self.args = _args
        self.base_dir = self.args.base_dir
        self.scribble_dir = os.path.join(self.base_dir, 'annotations')
        self.split = split
        self.fstroke = fstroke
        self.bstroke = bstroke
        self.instance = instance

        self.labels = {}
        with open(os.path.join(self.scribble_dir, 'annotations.pkl'), 'rb') as f:
            self.labels = pickle.load(f)

        self.images = list(self.labels.keys())

        self.perm = list(range(len(self.images)))   # random permutation of images and depths
        random.shuffle(self.perm)

        if self.args.in_chan==6:
            with open('normal_paths.pkl', 'rb') as f:
                self.normal_paths = pickle.load(f)

        if self.split=='train':
            self.perm = self.perm[:int(0.9*len(self.perm))]
        elif self.split=='val':
            self.perm = self.perm[int(0.9*len(self.perm)):]
        elif self.split=='test':
            pass    # base_dir should be the separate test folder
        else:
            raise ValueError("Only 'train', 'val', and 'test' are currently implemented as splits.")

    def __len__(self):
        return len(self.perm)
    
    def __getitem__(self, index):
        im_name = self.images[self.perm[index]]

        # Replacing keys. See the comment in FixedImageDataset.__getitem__ for details.
        im = Image.open(im_name.replace('img_corr', 'image'))

        # Draw labels to create label mask.
        fg_labels, bg_labels = self.labels[im_name]
        labels = np.ones((432, 540, 3))*255
        for i, l in enumerate(fg_labels):
            if len(l) > 0:
                cv2.polylines(labels, np.array([l]), False, ((i+1 if self.instance else 1), 0, 0), self.fstroke)
        for l in bg_labels:
            if len(l) > 0:
                cv2.polylines(labels, np.array([l]), False, (0, 0, 0), self.bstroke)

        labels = cv2.resize(labels, im.size)
        labels[labels>1] = 255
        labels = labels[:,:,0].astype(np.uint8)
        
        if self.split=='test':
            sample = im
        else:
            sample = {'image':im, 'label':Image.fromarray(labels)}
            if self.args.in_chan==4:
                # sample['depth'] = Image.fromarray(np.load(self.depths[self.perm[index]]))
                depth_arr = np.load(get_depth_from_image(im_name))
                sample['depth'] = Image.fromarray(depth_arr)
                # print(depth_arr.min(), depth_arr.max())
            if self.args.in_chan==6:
                normal_arr = np.load(self.normal_paths[im_name])
                sample['normal'] = normal_arr
            
        if self.split == "train":
            return self.transform_tr(sample)
        else:
            return self.transform_val(sample)

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            RandomHorizontalFlip(),
            RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size),
            RandomGaussianBlur(),
            Normalize(mean=MEAN, std=STDEV),
            ToTensor()])

        return composed_transforms(sample)

    def transform_val(self, sample):
        composed_transforms = transforms.Compose([
            FixScaleCrop(crop_size=self.args.crop_size),
            Normalize(mean=MEAN, std=STDEV),
            ToTensor()])

        return composed_transforms(sample)
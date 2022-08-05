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

# MEAN = [0.55615947, 0.6482298, 0.84056612]
# STDEV = [0.15693977, 0.16230874, 0.15292975]
MEAN = [0.485, 0.456, 0.406]
STDEV = [0.229, 0.224, 0.225]

def make_data_loader(*args, **kwargs):
    train_set = FoldSegmentation(args, split='train', bstroke=15)
    val_set = FoldSegmentation(args, split='val', bstroke=15)
    args = args[0]

    num_class = train_set.NUM_CLASSES
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
    test_loader = None

    return train_loader, val_loader, test_loader, num_class

class FoldSegmentation(Dataset):
    NUM_CLASSES = 2
    def __init__(
        self,
        _args,
        split='train',
        fstroke=5,
        bstroke=9,
    ):
        super().__init__()
        if isinstance(_args, tuple):
                self.args = _args[0]
        else:
            self.args = _args
        self.base_dir = self.args.base_dir
        self.scribble_dir = os.path.join(self.base_dir, 'annotations')
        self.split = split
        self.fstroke = fstroke
        self.bstroke = bstroke

        self.labels = {}
        with open(os.path.join(self.scribble_dir, 'annotations.pkl'), 'rb') as f:
            self.labels = pickle.load(f)

        self.images = list(self.labels.keys())
        if self.split=='train':
            self.images = self.images[:int(0.9*len(self.images))]
        elif self.split=='val':
            self.images = self.images[int(0.9*len(self.images)):]
        elif self.split=='test':
            pass    # base_dir should be the separate test folder
        else:
            raise ValueError("Only 'train', 'val', and 'test' are currently implemented as splits.")

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        im_name = self.images[index]
        im = Image.open(im_name)
        fg_labels, bg_labels = self.labels[im_name]
        labels = np.ones((432, 540, 3))*255
        for l in fg_labels:
            if len(l) > 0:
                cv2.polylines(labels, np.array([l]), False, (1, 0, 0), self.fstroke)
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

        if self.split == "train":
            return self.transform_tr(sample)
        elif self.split == 'val':
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
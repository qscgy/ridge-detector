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

def make_data_loader(*args, **kwargs):
    train_set = FoldSegmentation(args, split='train')
    val_set = FoldSegmentation(args, split='val')
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
    ):
        super().__init__()
        if isinstance(_args, tuple):
                self.args = _args[0]
        else:
            self.args = _args
        self.base_dir = self.args.base_dir
        self.scribble_dir = os.path.join(self.base_dir, 'annotations')
        self.split = split

        self.images = glob.glob(os.path.join(self.base_dir, '*.jpg'))
        self.images.extend(glob.glob(os.path.join(self.base_dir, '*.png')))
        self.images.extend(glob.glob(os.path.join(self.base_dir, '*.jpeg')))

        self.images = natsorted(self.images)
        if self.split=='train':
            self.images = self.images[:int(0.9*len(self.images))]
        elif self.split=='val':
            self.images = self.images[int(0.9*len(self.images)):]
        else:
            raise ValueError("Only 'train' and 'val' are currently implemented as splits.")

        self.fg_labels = {}
        with open(os.path.join(self.scribble_dir, 'folds.pkl'), 'rb') as f:
            self.fg_labels = pickle.load(f)
        
        self.bg_labels = {}
        with open(os.path.join(self.scribble_dir, 'backgrounds.pkl'), 'rb') as f:
            self.bg_labels = pickle.load(f)
        
        # print(len(self.images))
        # print(len(self.labels.keys()))
        # assert len(self.images) <= len(self.labels.keys())
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        im_name = self.images[index]
        im = Image.open(im_name)
        im_labels = self.fg_labels[im_name] if im_name in self.fg_labels else []
        labels = np.ones((432, 540, 3))*255
        for l in im_labels:
            if len(l) > 0:
                cv2.polylines(labels, np.array([l]), False, (1, 0, 0), 5)

        labels = cv2.resize(labels, im.size)
        labels[labels>1] = 255

        im_labels = self.bg_labels[im_name] if im_name in self.bg_labels else []
        for l in im_labels:
            if len(l) > 0:
                cv2.polylines(labels, np.array([l]), False, (0, 0, 0), 2)
        labels = labels[:,:,0].astype(np.uint8)

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
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensor()])

        return composed_transforms(sample)

    def transform_val(self, sample):
        composed_transforms = transforms.Compose([
            FixScaleCrop(crop_size=self.args.crop_size),
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensor()])

        return composed_transforms(sample)
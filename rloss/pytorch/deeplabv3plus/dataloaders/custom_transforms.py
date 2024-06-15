import torch
import random
import numpy as np
import cv2

from PIL import Image, ImageOps, ImageFilter
from scipy.ndimage import zoom

class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        depth = sample['depth'] if 'depth' in sample else None
        normal = sample['normal'] if 'normal' in sample else None

        img = np.array(img).astype(np.float32)
        mask = np.array(mask).astype(np.float32)
        img /= 255.0
        img -= self.mean
        img /= self.std

        out = {'image': img,
                'label': mask}
        if depth:
            out['depth'] = depth
        if normal is not None:
            out['normal'] = normal
        return out

class NormalizeImage(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        img = np.array(img).astype(np.float32)
        img /= 255.0
        img -= self.mean
        img /= self.std

        return img


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = sample['image']
        mask = sample['label']
        depth = sample['depth'] if 'depth' in sample else None
        normal = sample['normal'] if 'normal' in sample else None

        img = np.array(img).astype(np.float32).transpose((2, 0, 1))
        mask = np.array(mask).astype(np.float32)

        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).float()
        if depth:
            depth = torch.from_numpy(np.array(depth).astype(np.float32))
            depth.unsqueeze_(0)
        if normal is not None:
            normal = torch.from_numpy(normal)
            normal = torch.permute(normal, (2, 0, 1))

        out = {'image': img,
                'label': mask}
        if depth is not None:
            out['depth'] = depth
        if normal is not None:
            out['normal'] = normal
        return out

class ToTensorImage(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, img):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = np.array(img).astype(np.float32).transpose((2, 0, 1))
        img = torch.from_numpy(img).float()
        return img

class RandomHorizontalFlip(object):
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        depth = sample['depth'] if 'depth' in sample else None
        normal = sample['normal'] if 'normal' in sample else None
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            if 'depth' in sample:
                depth = depth.transpose(Image.FLIP_LEFT_RIGHT)
            if 'normal' in sample:
                # normal = normal.transpose(Image.FLIP_LEFT_RIGHT)
                normal = np.fliplr(normal)
        out = {'image': img,
                'label': mask}
        if depth:
            out['depth'] = depth
        if normal is not None:
            out['normal'] = normal
        return out


class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        rotate_degree = random.uniform(-1*self.degree, self.degree)
        img = img.rotate(rotate_degree, Image.BILINEAR)
        mask = mask.rotate(rotate_degree, Image.NEAREST)

        return {'image': img,
                'label': mask}


class RandomGaussianBlur(object):
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        depth = sample['depth'] if 'depth' in sample else None
        normal = sample['normal'] if 'normal' in sample else None

        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=random.random()))

        out = {'image': img,
                'label': mask}
        if depth:
            out['depth'] = depth
        if normal is not None:
            out['normal'] = normal
        return out



class RandomScaleCrop(object):
    '''
    The convention is to pad 255 and ignore the padded region. 
    However, in scribble-annotated images, we need to distinguish ignore region 
    and padded region for our loss function. So fill is 254 for padding.
    '''
    def __init__(self, base_size, crop_size, fill=254):
        self.base_size = base_size
        self.crop_size = crop_size
        self.fill = fill

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        depth = sample['depth'] if 'depth' in sample else None
        normal = sample['normal'] if 'normal' in sample else None
        # random scale (short edge)
        short_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        w, h = img.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        if depth:
            depth = depth.resize((ow, oh), Image.BILINEAR)
        if normal is not None:
            # normal = normal.resize((ow, oh), Image.BILINEAR)
            normal = zoom(normal, (oh*1.0/h, ow*1.0/w, 1), order=1)
        # pad crop
        if short_size < self.crop_size:
            padh = self.crop_size - oh if oh < self.crop_size else 0
            padw = self.crop_size - ow if ow < self.crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=self.fill)
            if depth:
                depth = ImageOps.expand(depth, border=(0, 0, padw, padh), fill=0)
            if normal is not None:
                # normal = ImageOps.expand(normal, border=(0, 0, padw , padh), fill=self.fill)
                normal = np.pad(normal, ((0, padh), (0, padw), (0,0)), mode='constant', constant_values=0)
        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - self.crop_size)
        y1 = random.randint(0, h - self.crop_size)
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        if depth:
            depth = depth.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        if normal is not None:
            # normal = normal.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
            normal = normal[y1:y1+self.crop_size,x1:x1+self.crop_size]

        out = {'image': img,
                'label': mask}
        if depth:
            out['depth'] = depth
        if normal is not None:
            out['normal'] = normal
        return out


class FixScaleCrop(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        depth = sample['depth'] if 'depth' in sample else None
        normal = sample['normal'] if 'normal' in sample else None

        w, h = img.size
        if w > h:
            oh = self.crop_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = self.crop_size
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        if depth:
            depth = depth.resize((ow, oh), Image.BILINEAR)
        if normal is not None:
            normal = zoom(normal, (oh*1.0/h, ow*1.0/w, 1), order=1)
        # center crop
        w, h = img.size
        x1 = int(round((w - self.crop_size) / 2.))
        y1 = int(round((h - self.crop_size) / 2.))
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        if depth:
            depth = depth.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        if normal is not None:
            normal = normal[y1:y1+self.crop_size,x1:x1+self.crop_size]

        out = {'image': img,
                'label': mask}
        if depth:
            out['depth'] = depth
        if normal is not None:
            out['normal'] = normal
        return out

class FixScaleCropImage(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, img):
        w, h = img.size
        if w > h:
            oh = self.crop_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = self.crop_size
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        # center crop
        w, h = img.size
        x1 = int(round((w - self.crop_size) / 2.))
        y1 = int(round((h - self.crop_size) / 2.))
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        return img



class FixedResize(object):
    def __init__(self, size):
        self.size = (size, size)  # size: (h, w)

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']

        assert img.size == mask.size

        img = img.resize(self.size, Image.BILINEAR)
        mask = mask.resize(self.size, Image.NEAREST)

        return {'image': img,
                'label': mask}


def denormalizeimage(images, mean=(0., 0., 0.), std=(1., 1., 1.)):
    """Denormalize tensor images with mean and standard deviation.
    Args:
        images (tensor): N*C*H*W
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    images = images.cpu().numpy()
    # N*C*H*W to N*H*W*C
    images = images.transpose((0,2,3,1))
    images *= std
    images += mean
    images *=255.0
    # N*H*W*C to N*C*H*W
    images = images.transpose((0,3,1,2))
    return torch.tensor(images)

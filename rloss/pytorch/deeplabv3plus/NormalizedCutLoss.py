import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import sys
sys.path.append("../wrapper/bilateralfilter/build/lib.linux-x86_64-3.9")
from bilateralfilter import bilateralfilter, bilateralfilter_batch
from .dataloaders.custom_transforms import denormalizeimage
import time
from multiprocessing import Pool
import multiprocessing
from itertools import repeat
import pickle

class NormalizedCutLossFunction(Function):

    @staticmethod
    def forward(ctx, images, segmentations, sigma_rgb, sigma_xy, ROIs):
        ctx.save_for_backward(segmentations)
        ctx.N, ctx.K, ctx.H, ctx.W = segmentations.shape
        
        ROIs = ROIs.unsqueeze_(1).repeat(1,ctx.K,1,1)
        segmentations = torch.mul(segmentations.cuda(), ROIs.cuda())
        ctx.ROIs = ROIs
        
        ncloss = 0.0
        images = images.numpy().flatten()
        segmentations = segmentations.cpu().numpy().flatten()
        AS = np.zeros(segmentations.shape, dtype=np.float32)
        bilateralfilter_batch(images, segmentations, AS, ctx.N, ctx.K, ctx.H, ctx.W, sigma_rgb, sigma_xy)
        ncloss -= np.dot(segmentations, AS)

        one = np.ones_like(segmentations)
        d = np.zeros_like(AS)
        bilateralfilter_batch(images, one, d, ctx.N, ctx.K, ctx.H, ctx.W, sigma_rgb, sigma_xy)
        ctx.d = d

        ncloss /= np.dot(d, segmentations)
    
        # averaged by the number of images
        ncloss /= ctx.N
        
        # ctx.AS = AS
        ctx.AS = np.reshape(AS, (ctx.N, ctx.K, ctx.H, ctx.W))
        return Variable(torch.tensor([ncloss]), requires_grad=True)

    @staticmethod
    def backward(ctx, grad_output):
        segmentations = ctx.saved_tensors[0].cpu()
        d_prime = torch.from_numpy(ctx.d)
        AS = torch.from_numpy(ctx.AS.flatten())

        segmentations = torch.flatten(segmentations)

        # print(f'AS device: {AS.device}')
        # print(f'segmentations device: {segmentations.device}')
        # print(f'd_prime device: {d_prime.device}')

        dS = d_prime @ segmentations
        
        grad_segmentation = segmentations * AS * d_prime
        grad_segmentation /= dS**2
        grad_segmentation -= (2*AS)/dS
        grad_segmentation *= grad_output
        grad_segmentation /= ctx.N
        grad_segmentation = grad_segmentation.cuda()
        grad_segmentation = grad_segmentation.reshape(ctx.AS.shape)
        grad_segmentation = torch.mul(grad_segmentation, ctx.ROIs.cuda())
        return None, grad_segmentation, None, None, None
    
class NormalizedCutLoss(nn.Module):
    def __init__(self, weight, sigma_rgb, sigma_xy, scale_factor):
        super(NormalizedCutLoss, self).__init__()
        self.weight = weight
        self.sigma_rgb = sigma_rgb
        self.sigma_xy = sigma_xy
        self.scale_factor = scale_factor
    
    def forward(self, images, segmentations, ROIs):
        scaled_images = F.interpolate(images,scale_factor=self.scale_factor) 
        scaled_segs = F.interpolate(segmentations,scale_factor=self.scale_factor,mode='bilinear',align_corners=False)
        scaled_ROIs = F.interpolate(ROIs.unsqueeze(1),scale_factor=self.scale_factor).squeeze(1)
        return self.weight*NormalizedCutLossFunction.apply(
                scaled_images, scaled_segs, self.sigma_rgb, self.sigma_xy*self.scale_factor, scaled_ROIs)
    
    def extra_repr(self):
        return 'sigma_rgb={}, sigma_xy={}, weight={}, scale_factor={}'.format(
            self.sigma_rgb, self.sigma_xy, self.weight, self.scale_factor
        )
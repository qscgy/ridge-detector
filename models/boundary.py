import torch
import torch.nn as nn
import torch.nn.functional as F
from ..util import plot_contour

class HeadBlock(nn.Module):
    def __init__(
        self,
        in_chan,
        out_chan,
        mid_chan=None,
        kernel=3,
        padding=1,
        activation=nn.ReLU,
        norm=nn.BatchNorm2d,
        dropout=0.1,
        stride=1
    ):
        super().__init__()
        self.in_chan = in_chan
        self.out_chan = out_chan
        if mid_chan is None:
            mid_chan = in_chan

        self.activation = activation()
        self.block = nn.Sequential(
            nn.Conv2d(in_chan, mid_chan, kernel_size=kernel, stride=stride, padding=padding),
            norm(mid_chan),
            activation(),
            nn.Dropout2d(p=dropout) if dropout else nn.Identity(),
            nn.Conv2d(mid_chan, out_chan, 1)
        )
    
    def forward(self, x):
        y = self.block(x)
        y = self.activation(y)
        return y

class BoundaryBranch(nn.Module):
    def __init__(
        self,
        backbone_out_chan,
        order=3,
        stride=1,
        norm=nn.BatchNorm2d,
    ):
        super().__init__()
        self.location_head = HeadBlock(
            backbone_out_chan,
            2,
            mid_chan=backbone_out_chan,
            kernel=7,
            padding=3,
            stride=stride,
            norm=norm,
        )
        self.param_head = HeadBlock(
            backbone_out_chan,
            order * 4,
            mid_chan=backbone_out_chan,
            kernel=7,
            padding=3,
            stride=stride,
            norm=norm,
        )
        self.order = order

    def forward(self, x):
        locations = self.location_head(x)
        params = self.param_head(x)
        
        mask = plot_contour(
            locations[...,0], locations[...,1],
            params[:self.order],
            params[self.order:self.order*2],
            params[self.order*2:self.order*3],
            params[self.order*3],
            x.shape[-2:],
        )

        return mask
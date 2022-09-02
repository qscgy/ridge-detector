import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def plot_contour(a0, c0, a, b, c, d, outsize):
    n = torch.arange(1,a.shape[-1]+1, device='cuda:0')
    t = torch.arange(0,1,0.0001, device='cuda:0').reshape(-1,1)
    sins = torch.sin(2*np.pi*n*t).T
    coss = torch.cos(2*np.pi*n*t).T

    X = a0 + torch.sum(a[...,None]*sins + b[...,None]*coss, axis=-2)
    Y = c0 + torch.sum(c[...,None]*sins + d[...,None]*coss, axis=-2)
    X = X.long()
    Y = Y.long()

    im = torch.zeros((*a.shape[:-1], *outsize), dtype=torch.float32, requires_grad=True)
    im2 = im.clone()
    for h in range (a.shape[-4]):
        for i in range(a.shape[-3]):
            for j in range(a.shape[-2]):
                im2[h,i, j, Y[h,i,j], X[h,i,j]] = 1
    mask = in_out(im2, a.shape[:-1]) * in_out(im2, a.shape[:-1], True)
    mask = mask + in_out(im2.swapaxes(-2,-1), a.shape[:-1]).swapaxes(-2,-1) * in_out(im2.swapaxes(-2,-1), a.shape[:-1], True).swapaxes(-2,-1)
    mask = mask.sum((1,2))>0


    return mask

def in_out(im, par_size, flip=False):
    if flip:
        im = torch.flip(im, (-2,))
    mask = im.clone()
    crossings = torch.zeros(*par_size, mask.shape[-1])
    for i in range(1, mask.shape[-2]-1):
        crossings += ((im[...,i,:]-im[...,i-1,:]) > 0)
        mask[...,i,:] = (crossings%2==1)

    if flip:
        mask = torch.flip(mask, (-2,))
    return mask

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
        print(locations.shape)
        print(params.shape)
        
        mask = plot_contour(
            locations[:,0], locations[:,1],
            params[:self.order],
            params[self.order:self.order*2],
            params[self.order*2:self.order*3],
            params[self.order*3:],
            x.shape[-2:],
        )

        return mask

def fill_fourier(z0, coeffs, dt, device='cpu'):
    order = coeffs.shape[1]//4
    a0 = z0[:,0]
    c0 = z0[:,1]
    a = coeffs[:,:order]
    b = coeffs[:,order:order*2]
    c = coeffs[:,order*2:order*3]
    d = coeffs[:,order*3:]

    # print(f'a shape: {a.shape}')
    n = torch.arange(1, a.shape[1]+1, device=device).reshape(1,-1)
    t = torch.arange(0,1+dt,dt, device=device).reshape(-1,1)
    # print(f'n shape: {n.shape}')
    # print(f't shape: {t.shape}')
    sins = torch.sin(2*np.pi*n*t)[...,None,None]
    coss = torch.cos(2*np.pi*n*t)[...,None,None]
    # print(f'sins shape: {sins.shape}')

    X = a0 + torch.sum(a*sins + b*coss, axis=1)
    Y = c0 + torch.sum(c*sins + d*coss, axis=1)
    # print(f'X shape: {X.shape}')

    # Derivatives
    Xp = torch.sum(2*n[...,None,None]*np.pi*(a*coss - b*sins), axis=1)
    Yp = torch.sum(2*n[...,None,None]*np.pi*(c*coss - d*sins), axis=1)
    # print(f'Xp shape: {Xp.shape}')

    iY, iX = torch.meshgrid(torch.arange(50), torch.arange(50), indexing='ij')
    iX = iX[...,None, None,None].to(device)
    iY = iY[...,None,None,None].to(device)
    # print(f'iX shape: {iX.shape}')

    # Winding number computed using the Cauchy integral formula
    index = torch.sum((Xp+1.j*Yp)/(X-iX+1.0j*(Y-iY))*dt, axis=2)*1/(2*np.pi*1.j)
    # print(f'index shape: {index.shape}')

    return (index.abs() > 0.9)

if __name__=='__main__':
    import matplotlib.pyplot as plt

    dt = 0.01
    z0 = torch.tensor([15,30], device='cuda').reshape(1,-1,1,1)
    coeffs = torch.tensor([0,0,5,10,20,5,0,0], device='cuda').reshape(1,-1,1,1).expand(1,8,3,3)
    index = fill_fourier(z0, coeffs, dt, 'cuda')

    plt.imshow(index.abs().cpu().detach().numpy()[...,0,0])
    plt.show()


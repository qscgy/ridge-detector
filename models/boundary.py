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
        kernel=5,
        padding=1,
        activation=nn.ReLU,
        norm=nn.BatchNorm2d,
        dropout=0,
        stride=1
    ):
        super().__init__()
        self.in_chan = in_chan
        self.out_chan = out_chan
        if mid_chan is None:
            mid_chan = in_chan

        self.activation = activation()
        self.block = nn.Sequential(
            nn.Conv2d(in_chan, mid_chan, kernel_size=kernel, stride=stride, padding=0),
            # nn.MaxPool2d(2, stride=2),
            norm(mid_chan),
            activation(),
            nn.Dropout2d(p=dropout) if dropout else nn.Identity(),
            nn.Conv2d(mid_chan, out_chan, kernel_size=3, padding=0),
            nn.AvgPool2d(2, stride=2),
            # nn.MaxPool2d(2, stride=2),
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
            4,
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
        # mask = fill_fourier(locations, params, 0.01)
        mask = fill_arc_v2(locations, params)

        return mask, params

def fill_fourier(z0, coeffs, dt):
    device = coeffs.device
    order = coeffs.shape[1]//4
    coeffs = coeffs.unsqueeze(1)
    a0 = z0[:,None,0]
    c0 = z0[:,None,1]
    a = coeffs[:,:,:order]
    b = coeffs[:,:,order:order*2]
    c = coeffs[:,:,order*2:order*3]
    d = coeffs[:,:,order*3:]

    # print(f'a shape: {a.shape}')
    # print(f'a0 shape: {a0.shape}')
    n = torch.arange(1, a.shape[-3]+1, device=device).reshape(1,-1)
    t = torch.arange(0,1+dt,dt, device=device).reshape(-1,1)
    # print(f'n shape: {n.shape}')
    # print(f't shape: {t.shape}')
    sins = torch.sin(2*np.pi*n*t)[...,None,None]
    coss = torch.cos(2*np.pi*n*t)[...,None,None]
    # print(f'sins shape: {sins.shape}')

    X = a0 + torch.sum(a*sins + b*coss, axis=-3)
    Y = c0 + torch.sum(c*sins + d*coss, axis=-3)
    # print(f'X shape: {X.shape}')

    # Derivatives
    Xp = torch.sum(2*n[...,None,None]*np.pi*(a*coss - b*sins), axis=-3)
    Yp = torch.sum(2*n[...,None,None]*np.pi*(c*coss - d*sins), axis=-3)
    # print(f'Xp shape: {Xp.shape}')
    # plt.plot(X[0,:,0,0].cpu(), Y[0,:,0,0].cpu())
    # plt.show()

    coords = torch.linspace(-1, 1, 53)
    iY, iX = torch.meshgrid(coords, coords, indexing='ij')
    iX = iX[...,None,None,None,None].to(device)
    iY = iY[...,None,None,None,None].to(device)
    # print(f'iX shape: {iX.shape}')

    # Winding number computed using the Cauchy integral formula
    index = torch.sum((Xp+1.j*Yp)/(X-iX+1.0j*(Y-iY))*dt, axis=-3)*1/(2*np.pi*1.j)
    index = (index.real.abs() > 0.9).float()
    # index = index.real
    # print(f'index shape: {index.shape}')

    return index, X, Y

def fill_arc(center, params, device='cuda'):
    # theta >= 0 or this may not work
    R1 = params[:,0].unsqueeze(1)
    R2 = params[:,1].unsqueeze(1)
    t1 = params[:,2].unsqueeze(1)
    t2 = params[:,3].unsqueeze(1)
    ii = torch.linspace(-1,1,53, device=device)
    ix, iy = torch.meshgrid(ii, ii, indexing='xy')
    iy = torch.flipud(iy)
    ix = ix.reshape(*ix.shape,1,1,1,1)
    iy = iy.reshape(*iy.shape,1,1,1,1)
    ix = ix + center[:,0].unsqueeze(1)
    iy = iy + center[:,1].unsqueeze(1)
    rs = torch.sqrt(ix**2 + iy**2)
    thetas = torch.atan2(iy, ix)
    thetas = (thetas-t1)%(2*np.pi)
    mask = (thetas<=t2)
    dist = (R1-rs).abs()
    dist[~mask] = min_dist(ix, iy, R1, t1, t2)[~mask]
    dist = (dist<=R2)
    dist.squeeze_()
    dist_all = (torch.sum(dist, axis=(-1, -2))>=1).float()

    return dist_all

def fill_arc_v2(locs, params, device='cuda', outsize=216):
    # locs is now the point halfway along the arc
    R1 = params[:,0].unsqueeze(1)
    R2 = params[:,1].unsqueeze(1)
    alpha = params[:,2].unsqueeze(1)    # angle of unit tangent vector at center
    theta = params[:,3].unsqueeze(1)

    t1 = (alpha-np.pi/2-theta)%(2*np.pi)
    t2 = (2*theta)%(2*np.pi)
    print(theta[0,0,0,0]/np.pi)

    center = locs.clone()
    center[:,0] = (locs[:,0]-R1*torch.cos(alpha-np.pi/2))[:,0]
    center[:,1] = (locs[:,1]-R1*torch.sin(alpha-np.pi/2))[:,0]

    ii = torch.linspace(-1,1,outsize,device=device)
    ix, iy = torch.meshgrid(ii, ii, indexing='xy')
    iy = torch.flipud(iy)
    ix = ix.reshape(*ix.shape,1,1,1,1)
    iy = iy.reshape(*iy.shape,1,1,1,1)
    ix = ix - center[:,0].unsqueeze(1)
    iy = iy - center[:,1].unsqueeze(1)
    rs = torch.sqrt(ix**2 + iy**2)
    thetas = torch.atan2(iy, ix)
    thetas = (thetas-t1)%(2*np.pi)
    mask = (thetas<=t2)
    dist = (R1-rs).abs()
    dist[~mask] = min_dist(ix, iy, R1, t1, t2)[~mask]
    dist = (dist<=R2)
    dist.squeeze_()
    dist_all = (torch.sum(dist, axis=(-1, -2))>=1).float()

    return dist_all

def min_dist(ix, iy, R1, t1, t2):
    x1 = R1*torch.cos(t1)
    y1 = R1*torch.sin(t1)
    x2 = R1*torch.cos(t1+t2)
    y2 = R1*torch.sin(t1+t2)
    return torch.minimum(torch.sqrt((ix-x1)**2 + (iy-y1)**2), torch.sqrt((ix-x2)**2 + (iy-y2)**2))

if __name__=='__main__':
    import matplotlib.pyplot as plt
    from matplotlib import patches

    dt = 0.01
    z0 = torch.tensor([0,0], device='cuda').reshape(1,-1,1,1)
    coeffs = torch.randn(4,8,3,3, device='cuda')/4
    # index, xs, ys = fill_fourier(z0, coeffs, dt)

    center = torch.rand(4,2,3,3).abs().cuda()*1.4-0.7
    # center = torch.zeros(4,2,3,3).cuda()
    R1 = torch.rand(4,1,3,3).cuda()
    R2 = torch.rand(4,1,3,3).abs().cuda()*0.1
    alpha = torch.rand(4,1,3,3).cuda()*np.pi*2
    theta = torch.rand(4,1,3,3).cuda()*np.pi
    params = torch.cat((R1, R2, alpha, theta), 1)
    mask = fill_arc_v2(center, params)
    print(mask.shape)
    
    fig, ax = plt.subplots(1,1,figsize=(7,7))
    # for i in range(3):
    #     for j in range(3):
            # ax[i][j].imshow(index[:,:,0].cpu().detach().numpy()[...,i,j], extent=[-1,1,-1,1])
    ax.imshow(mask[:,:,0].cpu().detach().numpy(), extent=[-1,1,-1,1])
    ax.set_xlim(-1,1)
    ax.set_ylim(-1,1)
    ax.scatter(*center[0].detach().cpu().reshape(2,9))
    # arc = patches.Arc((0,0), 2, 2, 0, theta[0,0,i,j].cpu().numpy()*180/np.pi, (theta[0,0,i,j]+theta[0,1,i,j]).cpu().numpy()*180/np.pi, linewidth=5)
            # ax[i][j].add_patch(arc)
            # ax[i][j].plot(xs[0,:,i,j].cpu().detach().numpy(), ys[0,:,i,j].cpu().detach().numpy())
    plt.show()


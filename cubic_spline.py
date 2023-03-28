# from https://gist.github.com/chausies/c453d561310317e7eda598e229aea537

import torch
import torch.nn as nn

class Interpolator(nn.Module):
  def __init__(self, n_pts=9, max=1) -> None:
    super().__init__()
    assert n_pts>1
    self.n_pts = n_pts
    self.x_ctl = nn.Parameter(torch.linspace(0, max, n_pts))
    self.y_ctl = nn.Parameter(torch.linspace(0, max, n_pts))
    A = torch.tensor([
      [1, 0, -3, 2],
      [0, 1, -2, 1],
      [0, 0, 3, -2],
      [0, 0, -1, 1]
      ], dtype=torch.float)
    self.register_buffer('A', A)
  
  def __str__(self) -> str:
    return str(self.x_ctl)+'\n'+str(self.y_ctl)
  
  def forward(self, xs):
    m = (self.y_ctl[1:] - self.y_ctl[:-1])/(self.x_ctl[1:] - self.x_ctl[:-1])
    m = torch.cat([m[[0]], (m[1:] + m[:-1])/2, m[[-1]]])
    I = torch.searchsorted(self.x_ctl[1:], xs)
    dx = (self.x_ctl[I+1]-self.x_ctl[I])
    hh = self.h_poly((xs-self.x_ctl[I])/dx)
    return hh[0]*self.y_ctl[I] + hh[1]*m[I]*dx + hh[2]*self.y_ctl[I+1] + hh[3]*m[I+1]*dx

  def h_poly(self, t):
    tt = torch.ones(4, t.shape[-1])
    tt = tt.to(self.A.device)
    tt[1] = tt[0].clone()*t
    tt[2] = tt[1].clone()*t
    tt[3] = tt[2].clone()*t
    out = self.A @ tt
    return out

  def H_poly(self, t):
    tt = torch.zeros(4,1).to(self.A.device)
    tt[0] = t
    for i in range(1, 4):
      tt[i] = tt[i-1]*t*i/(i+1)
    return self.A @ tt

def interp_func(x, y):
  "Returns integral of interpolating function"
  if len(y)>1:
    m = (y[1:] - y[:-1])/(x[1:] - x[:-1])
    m = torch.cat([m[[0]], (m[1:] + m[:-1])/2, m[[-1]]])
  def f(xs):
    if len(y)==1: # in the case of 1 point, treat as constant function
      return y[0] + torch.zeros_like(xs)
    I = torch.searchsorted(x[1:], xs)
    dx = (x[I+1]-x[I])
    hh = h_poly((xs-x[I])/dx)
    return hh[0]*y[I] + hh[1]*m[I]*dx + hh[2]*y[I+1] + hh[3]*m[I+1]*dx
  return f

def interp(x, y, xs):
    if len(y)>1:
        m = (y[1:] - y[:-1])/(x[1:] - x[:-1])
        m = torch.cat([m[[0]], (m[1:] + m[:-1])/2, m[[-1]]])
    if len(y)==1: # in the case of 1 point, treat as constant function
        return y[0] + torch.zeros_like(xs)
    I = torch.searchsorted(x[1:], xs)
    dx = (x[I+1]-x[I])
    hh = h_poly((xs-x[I])/dx)
    return hh[0]*y[I] + hh[1]*m[I]*dx + hh[2]*y[I+1] + hh[3]*m[I+1]*dx

def integ_func(x, y):
  "Returns interpolating function"
  if len(y)>1:
    m = (y[1:] - y[:-1])/(x[1:] - x[:-1])
    m = torch.cat([m[[0]], (m[1:] + m[:-1])/2, m[[-1]]])
    Y = torch.zeros_like(y)
    Y[1:] = (x[1:]-x[:-1])*(
        (y[:-1]+y[1:])/2 + (m[:-1] - m[1:])*(x[1:]-x[:-1])/12
        )
    Y = Y.cumsum(0)
  def f(xs):
    if len(y)==1:
      return y[0]*(xs - x[0])
    I = torch.searchsorted(x[1:], xs)
    dx = (x[I+1]-x[I])
    hh = H_poly((xs-x[I])/dx)
    return Y[I] + dx*(
        hh[0]*y[I] + hh[1]*m[I]*dx + hh[2]*y[I+1] + hh[3]*m[I+1]*dx
        )
  return f

def integ(x, y, xs):
  return integ_func(x,y)(xs)

if __name__ == "__main__":
    import matplotlib.pylab as P # for plotting
    x = torch.linspace(0, 6, 7)
    y = x.sin()
    xs = torch.linspace(0, 6, 101)
    ys = interp(x, y, xs)
    Ys = integ(x, y, xs)
    P.scatter(x, y, label='Samples', color='purple')
    P.plot(xs, ys, label='Interpolated curve')
    P.plot(xs, xs.sin(), '--', label='True Curve')
    P.plot(xs, Ys, label='Spline Integral')
    P.plot(xs, 1-xs.cos(), '--', label='True Integral')
    P.legend()
    P.show()
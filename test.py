import torch
import zejun_cuda
import time 

def trilinear_interpolation_py(feats, points):
    """
    Inputs:
        feats: (N, 8, F)
        points: (N, 3) local coordinates in [-1, 1]
    
    Outputs:
        feats_interp: (N, F)
    """
    u = (points[:, 0:1]+1)/2
    v = (points[:, 1:2]+1)/2
    w = (points[:, 2:3]+1)/2
    a = (1-v)*(1-w)
    b = (1-v)*w
    c = v*(1-w)
    d = 1-a-b-c

    feats_interp = (1-u)*(a*feats[:, 0] +
                          b*feats[:, 1] +
                          c*feats[:, 2] +
                          d*feats[:, 3]) + \
                       u*(a*feats[:, 4] +
                          b*feats[:, 5] +
                          c*feats[:, 6] +
                          d*feats[:, 7])
    
    return feats_interp


N = 65536; F=256
features = torch.rand(N,8,F,device='cuda')
points = torch.rand(N,3,device='cuda')

t = time.time()
out_cuda = zejun_cuda.trilinear_interpolation(features, points)
torch.cuda.synchronize()
print(time.time()-t)
print(out_cuda)
print(out_cuda.shape)
t = time.time()
out_py = trilinear_interpolation_py(features, points)
torch.cuda.synchronize()
print(time.time()-t)
print(out_py)
print(out_py.shape)

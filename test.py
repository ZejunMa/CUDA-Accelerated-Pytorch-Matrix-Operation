import torch
import zejun_cuda

features = torch.ones(2).to('cuda')
point = torch.zeros(2).to('cuda')
print(features.device)
out = zejun_cuda.trilinear_interpolation(features, point)
print(out)

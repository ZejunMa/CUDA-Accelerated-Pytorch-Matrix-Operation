import torch
import zejun_torch_cuda

features = torch.ones(2)
point = torch.zeros(2)

features.requires_grad = True
out = zejun_torch_cuda.trilinear_interpolation(features, point)
print(out.requires_grad)
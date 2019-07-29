import torch

x = torch.arange(8).view(2, 2, 2)

print(x)

x = x.flip([2])

print('\n',x)
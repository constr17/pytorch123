import torch

batch_size = 10
features = 25
x = torch.randn((batch_size, features))

print(x[0].shape) # x[0,:]
print(x[:,0].shape) # x[:,0]
x[2, 1] = 100
print(x[2, 0:10]) # 0:10 --> [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

x = torch.arange(10)
indices = [2, 3, 8]
print(x[indices]) # [2, 3, 8]

x1 = torch.rand((3, 5))
rows = torch.tensor([1, 0])
cols = torch.tensor([4, 0])
print(x1[rows, cols].shape)

x = torch.arange(10)
print(x[(x < 2) | (x > 8)])
print(x[x.remainder(2) == 0])
print(torch.where(x <= 5, x, x*2))
print(torch.tensor([0,0,1,1,1,2,2,2,3,4]).unique())
print(x1.numel())

x = torch.arange(9)
x33 = x.view(3, 3)
x33 = x.reshape(3, 3)
print(x33, x33.t(), x33.t().contiguous().view(9))

x1 = torch.rand((2, 3))
x2 = torch.rand((2, 3))
print(torch.cat((x1, x2), dim=0).shape, torch.cat((x1, x2), dim=1).shape, x1.view(-1).shape)

batch_size = 64
x = torch.rand((batch_size, 2, 5))
z = x.view((batch_size, -1))
print(z.shape)
z = x.permute(0, 2, 1)
print(z.shape)
x = torch.arange(10)
print(x.unsqueeze(0).shape)
print(x.unsqueeze(1).shape)

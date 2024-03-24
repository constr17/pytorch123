import torch
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'
my_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]],
                         dtype=torch.float32,
                         device=device,
                         requires_grad=True)

print(my_tensor)
print(my_tensor.dtype, my_tensor.device, my_tensor.shape, my_tensor.requires_grad)

x = torch.empty(size=(3,3))
x = torch.zeros((3,3))
x = torch.ones((3,3))
x = torch.rand((3,3))
x = torch.eye(3, 3)
x = torch.arange(start=0, end=5, step=1)
x = torch.linspace(start=0.1, end=1, steps=10)
x = torch.empty(size=(1,5)).normal_(mean=0, std=1)
x = torch.empty(size=(1,5)).uniform_(0, 1)
x = torch.diag(torch.ones(3))
print(x)

tensor = torch.arange(4)
print(tensor.bool())
print(tensor.short())
print(tensor.long())
print(tensor.float())
print(tensor.half())

np_array = np.eye(5)
tensor = torch.from_numpy(np_array)
np_array_back = tensor.numpy()
print(np_array_back)

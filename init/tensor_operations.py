# PYTORCH COMMON MISTAKES - How To Save Time 🕒 (https://youtu.be/O2wJ3tkc-TU?si=bMr1tc4cbiHCgg1e)
import torch

x = torch.tensor([[1, 2, 3], [4, 5, 6]])

print(x)

print(x.view((3,2)))  # Не транспонирование

print(x.permute(1, 0))  # Транспонирование
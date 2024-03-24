import torch

x = torch.tensor([1, 2, 3])
y = torch.tensor([9, 8, 7])

z1 = torch.empty(3)
torch.add(x, y, out=z1)
z2 = torch.add(x, y)
z = x + y
z = x - y
z = torch.true_divide(x, y)
print(z)

t = torch.zeros(3)
t.add_(x)
t += x
print(t)

z = x.pow(2)
z = x ** 2
print(z)

z = x > 1
print(z)

x1 = torch.rand((2, 5))
x2 = torch.rand((5, 3))
x3 = torch.mm(x1, x2) # 2x3
x3 = x1.mm(x2) # 2x3
print(x3)

matrix_exp = torch.rand(3, 3)
print(matrix_exp.matrix_power(3))
print(x * y, torch.dot(x,y))

batch = 32
n = 10
m = 20
p = 30
t1 = torch.rand((batch, n, m))
t2 = torch.rand((batch, m, p))
out_bmm = torch.bmm(t1, t2)
print(out_bmm)

sum_x = torch.sum(x, dim=0)
print(sum_x)
values, indexes = torch.max(x, dim=0)
values, indexes = torch.min(x, dim=0)
print(values, indexes)
abs_x = torch.abs(x)
print(abs_x)
z = torch.argmax(x, dim=0)
print(z)
mean_x = torch.mean(x.float(), dim=0)
print(mean_x)
z = torch.eq(x, y)
print(z)
z = torch.sort(x, dim=0, descending=True)
print(z.values)
z = torch.clamp(x, min=1, max=2)
print(z)

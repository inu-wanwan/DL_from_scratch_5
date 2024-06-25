import torch
import matplotlib.pyplot as plt

torch.manual_seed(0)
x = torch.rand(100, 1)
y = 2 * x + 5 + torch.rand(100, 1)

W = torch.zeros((1, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

def predict(x):
    return x @ W + b

def mse(x0, x1):
    diff = x0 - x1
    N = len(diff)
    return torch.sum(diff ** 2) / N

lr = 0.1
iters = 1000

for i in range(iters):
    y_hat = predict(x)
    loss = mse(y, y_hat)

    loss.backward()

    W.data -= lr * W.grad.data
    b.data -= lr * b.grad.data

    W.grad.data.zero_()
    b.grad.data.zero_()

    if i % 100 == 0:
        print(f'{i}: {loss.item()}')

print(loss.item())
print('=====')
print('W:', W.item())
print('b:', b.item())
import torch

def rosenbrock(x0, x1):
    return 100 * (x1 - x0 ** 2) ** 2 + (1 - x0) ** 2

x0 = torch.tensor(0.0, requires_grad=True)
x1 = torch.tensor(2.0, requires_grad=True)

lr = 0.001
iters = 10000

for i in range(iters):
    if i % 1000 == 0:
        print(f'{i}: {x0.item()}, {x1.item()}')

    y = rosenbrock(x0, x1)

    y.backward()

    # update x0 and x1
    x0.data -= lr * x0.grad.data
    x1.data -= lr * x1.grad.data

    # clear grad
    x0.grad.data.zero_()
    x1.grad.data.zero_()

print("=====")
print(f'{iters}: {x0.item()}, {x1.item()}')
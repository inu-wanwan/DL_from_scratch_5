import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

torch.manual_seed(0)
x = torch.rand(100, 1)
y = torch.sin(2 * torch.pi * x) + torch.rand(100, 1)

class Model(nn.Module):
    def __init__(self, input_size=1, hidden_seize=10, output_size=1):
        super().__init__()
        self.lenear1 = nn.Linear(input_size, hidden_seize)
        self.lenear2 = nn.Linear(hidden_seize, output_size)

    def forward(self, x):
        y = self.lenear1(x)
        y = F.sigmoid(y)
        y = self.lenear2(y)
        return y
    
lr = 0.2
iters = 10000

model = Model()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

for i in range(iters):
    y_pred = model(x)
    loss = F.mse_loss(y, y_pred)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if i % 1000 == 0:
        print(f'{i}: {loss.item()}')

print(loss.item())

# visualize
plt.scatter(x.detach().numpy(), y.detach().numpy(), s=10)
x = torch.linspace(0, 1, 100).reshape(-1, 1)
y = model(x).detach().numpy()
plt.plot(x, y, color='red')
plt.show()
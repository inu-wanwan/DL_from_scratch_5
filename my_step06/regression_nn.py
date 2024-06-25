import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = nn.Linear(10, 5)  # self.linearを定義

    def forward(self, x):
        return self.linear(x)  # self.linearを使用

# インスタンス化と使用
x = torch.rand(1, 100)
y = 5 + 2 * x + torch.rand(1, 100)

lr = 0.1
iters = 1000

model = Model()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

model(torch.tensor([1.0]))  # 一度forwardを実行しておく
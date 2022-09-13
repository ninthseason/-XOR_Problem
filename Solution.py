import random

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F


class XOR(nn.Module):
    def __init__(self):
        super().__init__()
        self.hid1 = nn.Linear(in_features=2, out_features=2)
        self.hid2 = nn.Linear(in_features=2, out_features=2)
        self.out = nn.Linear(in_features=2, out_features=1)
        nn.init.uniform_(self.hid1.weight, -1, 1)
        nn.init.uniform_(self.hid2.weight, -1, 1)
        nn.init.uniform_(self.out.weight, -1, 1)

    def forward(self, x):
        x = F.leaky_relu(self.hid1(x), inplace=True)
        x = F.leaky_relu(self.hid2(x), inplace=True)
        y = F.leaky_relu(self.out(x))
        return y


DEVICE = "cpu"

model = XOR()
criticizer = nn.MSELoss()
optimizer = optim.Adam(params=model.parameters(), lr=0.01)


def train():
    model.train()
    x = [[random.randint(0, 1), random.randint(0, 1)] for _ in range(100)]
    x = torch.tensor(x).to(DEVICE)
    y = (x[:, 0] ^ x[:, 1]).unsqueeze(1).to(torch.float)
    x = x.to(torch.float)
    y_hat = model(x)

    optimizer.zero_grad()
    loss = criticizer(y_hat, y)
    loss.backward()
    optimizer.step()


def evaluate():
    model.eval()
    x = [[0, 0],
         [0, 1],
         [1, 0],
         [1, 1]]
    print("操作数", x)
    x = torch.tensor(x, requires_grad=False).to(DEVICE)
    y = list((x[:, 0] ^ x[:, 1]).numpy())
    x = x.to(torch.float)
    y_hat = model(x).squeeze(1).detach().numpy()
    print("拟合后", y_hat)
    y_hat = list(map(lambda z: 1 if z >= 0.5 else 0, y_hat))
    print(y_hat)
    print(y)


for i in range(1000):
    train()
evaluate()

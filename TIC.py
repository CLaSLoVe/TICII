import torch
from torch import nn
import torch.nn.functional as F
from load_data import XplaneDataset, DataLoader

device = 'cuda:0'
# device = 'cpu'

class TIC(nn.Module):
    def __init__(self, num_sensors, delta_t, num_clusters):
        super().__init__()

        self.delta_t = delta_t
        self.num_sensors = num_sensors
        self.num_clusters = num_clusters

        # self.beta = nn.Parameter(torch.randn(1))
        self.mu = nn.Parameter(torch.randn(num_clusters, num_sensors, device=device))

        self.A = nn.Parameter(torch.randn((delta_t, num_clusters, num_sensors, num_sensors), device=device))

    def forward(self, X, lead=None):
        assert X.shape[-1] == self.num_sensors and X.shape[-2] == self.delta_t

        # spread X into multi clusters, and minus mu
        X = torch.stack([X] * self.num_clusters, dim=1)
        dX = X - torch.stack([self.mu] * self.delta_t, dim=1)
        dX = dX.reshape((X.shape[0], self.num_clusters, 1, -1))
        self.dX = dX

        # generate Theta
        mat = [[None for _ in range(self.delta_t)] for _ in range(self.delta_t)]
        for i in range(self.delta_t):
            for j in range(self.delta_t):
                if i >= j:
                    mat[i][j] = self.A[i - j].transpose(-1, -2)
                else:
                    mat[i][j] = self.A[j - i]
        self.mat = torch.cat([torch.cat(row, dim=2) for row in mat], dim=1)

        # calc lle
        P1 = dX @ self.mat @ dX.transpose(-1, -2)
        P2 = torch.slogdet(self.mat)[1]
        self.P1 = P1
        self.P2 = P2
        lle = -.5 * P1.reshape((P1.shape[0], self.num_clusters)) + P2.reshape(self.num_clusters)
        self.lle = lle

        return torch.softmax(lle, dim=-1)


criterion = nn.CrossEntropyLoss()
l1_lambda = 0.01


def loss(model, outputs, targets):
    # 计算交叉熵损失
    ce_loss = criterion(outputs, targets)

    # 计算L1正则化损失
    l1_loss = torch.norm(model.mat, p=1, dim=[1, 2]).mean()

    # 将两种损失结合起来
    res = ce_loss + l1_lambda * l1_loss

    return res


if __name__ == '__main__':
    batch_size = 8
    num_sensors = 12
    delta_t = 3
    num_clusters = 4
    num_epoch = 5

    dataset = XplaneDataset(delta_t)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    net = TIC(num_sensors, delta_t, num_clusters)
    net = net.to(device)
    updater = torch.optim.Adam(net.parameters(), lr=.001)

    for i in range(num_epoch):
        sum_loss = 0
        for batch in dataloader:
            X, y = batch
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(net, y_hat, y)
            updater.zero_grad()
            l.backward()
            updater.step()

            # 计算梯度范数并打印
            grad_norm = 0
            for p in net.parameters():
                grad_norm += torch.norm(p.grad) ** 2
            grad_norm = grad_norm ** (1 / 2)
            # print('Gradient norm:', grad_norm.item())

            sum_loss += l.item()
        wrong = 0
        for batch in dataloader:
            X, y = batch
            X, y = X.to(device), y.to(device)
            ans = torch.sum(abs(net(X) - y), dim=1)
            wrong += torch.count_nonzero(ans).item()

        print('epoch', i+1, ':', (1 - wrong/len(dataset.X))*100, '%')
        print('\tloss:', sum_loss/len(dataset.X))


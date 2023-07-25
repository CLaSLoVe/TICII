import torch
from torch import nn
import torch.nn.functional as F
from load_data import XplaneDataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib.colors as colors
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


class ADMM(torch.optim.Optimizer):
    def __init__(self, params, rho=1.0, lr=1e-3):
        defaults = dict(lr=lr)
        super().__init__(params, defaults)
        self.rho = rho

        self.z = [torch.zeros_like(p) for p in params]
        self.u = [torch.zeros_like(p) for p in params]

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        rho = self.rho
        for i, group in enumerate(self.param_groups):
            lr = group['lr']
            params = group['params']
            for j, p in enumerate(params):
                z = self.z[j]
                u = self.u[j]
                p.data = (z - u) / rho
                p.data = torch.sign(p.data) * torch.max(torch.abs(p.data) - lr / rho, torch.zeros_like(p.data))
                z.data = p.data + u
                u.data = u.data + p.data - z.data

        return loss


criterion = nn.CrossEntropyLoss()
l1_lambda = .01


def loss(model, outputs, targets):
    ce_loss = criterion(outputs, targets)
    l1_loss = torch.norm(model.mat, p=1, dim=[1, 2]).max()
    res = ce_loss + l1_lambda * l1_loss
    return res


def calculate_accuracy(model, dataloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data, labels in dataloader:
            outputs = model(data)
            _, predictions = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predictions == labels).sum().item()
    accuracy = correct / total
    return accuracy


if __name__ == '__main__':
    batch_size = 8
    num_sensors = 12
    delta_t = 3
    num_clusters = 4
    num_epoch = 1

    dataset = XplaneDataset(delta_t)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    net = TIC(num_sensors, delta_t, num_clusters)
    net = net.to(device)
    # updater = torch.optim.Adam(net.parameters(), lr=0.01)
    updater = ADMM(net.parameters(), rho=1.0)

    for i in range(num_epoch):
        sum_loss = 0
        correct = 0
        for batch in dataloader:
            X, y = batch
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(net, y_hat, y)
            updater.zero_grad()
            l.backward()
            updater.step()
            # break
        for batch in dataloader:
            X, y = batch
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            sum_loss += l.item()
            predictions = torch.argmax(y_hat, dim=1)
            labels = torch.argmax(y, dim=1)
            correct += (predictions == labels).sum().item()

        print('epoch', i+1, ':', (correct/len(dataset.X))*100, '%')
        print('\tloss:', sum_loss/len(dataset.X))

    # matrix=-abs(net.mat[0].detach().cpu().numpy())
    # cmap = 'gray'
    # norm = colors.Normalize(vmin=0, vmax=matrix.max())
    # plt.imshow(matrix, cmap=cmap, norm=norm)
    # # 显示颜色条
    # plt.colorbar()
    # # 显示图像
    # plt.show()

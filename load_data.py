import torch
from torch.utils.data import Dataset, DataLoader
import csv


class XplaneDataset(Dataset):
    def __init__(self, time_window=2):
        with open('data.csv', 'r') as f:
            X = list(csv.reader(f))
        self.X = torch.tensor([list(map(float, x)) for x in X])
        with open('one_hot.csv', 'r') as f:
            Y = list(csv.reader(f))
        self.y = torch.tensor([list(map(float, y)) for y in Y])
        self.time_window = time_window

        self.mean = torch.mean(self.X, dim=0)
        self.std = torch.std(self.X, dim=0)
        ind = self.std.nonzero().T.tolist()[0]
        self.X = self.X[:, ind]
        self.mean = self.mean[ind]
        self.std = self.std[ind]
        self.X = (self.X - self.mean)/self.std

    def __len__(self):
        return len(self.X) - self.time_window

    def __getitem__(self, index):
        X_window = self.X[index:index+self.time_window]
        y_window = self.y[index+self.time_window]

        return X_window, y_window


if __name__ == '__main__':
    dataset = XplaneDataset()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    for batch in dataloader:
        print(batch)
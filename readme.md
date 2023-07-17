# Toeplitz Inversed Covariance based Online Segment in Pytorch 🚀

This repository contains a PyTorch implementation of a Toeplitz Inversed Covariance based Online Segment (TICOS) model for time series data analysis. The TICOS model is a clustering-based method to learn a time-invariant representation of the time series data.

## Usage 📈

To use the TICOS model, you will need to provide a dataset of time series data. The `XplaneDataset` class in the `load_data.py` file provides an example dataset that you can use to test the model.

```python
from load_data import XplaneDataset
dataset = XplaneDataset(delta_t)
```

To train the TICOS model on your dataset, you can use the `TIC` and `Loss` classes provided in the `TIC.py` file. `TIC` is the main model class, which takes as input the number of sensors in the data, the time window size `delta_t`, and the number of clusters to use for clustering. `Loss` is a custom loss function that incorporates both cross-entropy loss and L1 regularization to encourage sparsity in the learned representation.

```python
from tic import TIC, Loss
net = TIC(num_sensors, delta_t, num_clusters)
loss = Loss(net)
```

You can then train the model using a PyTorch optimizer such as Adam or SGD, and iterate over your dataset using a PyTorch DataLoader.

```python
updater = torch.optim.Adam(net.parameters(), lr=.01)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
for epoch in range(num_epoch):
    for batch in dataloader:
        X, y = batch
        y_hat = net(X)
        l = loss(y_hat, y)
        updater.zero_grad()
        l.backward()
        updater.step()
```

## Acknowledgements 👏

This implementation is derived from the paper "Toeplitz Inverse Covariance-Based Clustering of Multivariate Time Series Data".

## License 📝

This code is released under the MIT License. See the LICENSE file for details.
import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l


# synthetic data
true_w = torch.tensor([3.0, -5.1, 6.8])
true_b = 2.2
num_examples = 2000
features, labels = d2l.synthetic_data(true_w, true_b, num_examples)

def load_array(data_arrays, batch_size, is_train=True):
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, is_train)

batch_size = 10
data_iter = load_array((features, labels), batch_size)

net = torch.nn.Sequential(torch.nn.Linear(3, 1))
net[0].weight.data.normal_(5, 1)
net[0].bias.data.fill_(9)

loss = torch.nn.MSELoss()

trainer = torch.optim.SGD(net.parameters(), lr=0.01)

num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X), y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')

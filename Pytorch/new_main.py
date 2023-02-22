import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
from tqdm.notebook import tqdm
from torch.utils.tensorboard import SummaryWriter
from network import Network
import numpy as np
EPOCHS = 5
train_set = torchvision.datasets.MNIST(
    root='./datasets/',
    train=True,
    download=True,
    transform=transforms.ToTensor()
)
test_set = torchvision.datasets.MNIST(
    root='./datasets/',
    train=False,
    download=True,
    transform=transforms.ToTensor()
)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=100, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=100)
channels = [16, 32]
kSizes = [5, 5]
model = Network(channels, kSizes)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
for epoch in range(EPOCHS):
    print('EPOCH {}:'.format(epoch + 1))
    for i, data in tqdm(train_loader):
        images, labels = data
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

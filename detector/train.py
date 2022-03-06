#!/usr/bin/env python
import argparse
import logging
import sys
import uuid

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import TileData


class ConvNet(nn.Module):
    """XXX
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(2496, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 28)  # XXX 28 == num classes

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, help='path to labelled data')
    parser.add_argument('--batch-size', type=int, default=4, help='data loader batch size')
    parser.add_argument('--epochs', type=int, default=2, help='number of epochs to train')
    parser.add_argument('--model', help='path to model on disk')
    args = parser.parse_args()

    data = TileData(path=args.data)
    logging.info(f'loaded {len(data)} samples')

    loader = DataLoader(data, batch_size=args.batch_size, shuffle=True)

    model = ConvNet()
    if args.model:
        logging.info(f'loading model from {args.model}')
        model.load_state_dict(torch.load(args.model))

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(args.epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(loader, 0):
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            pred = model(inputs)
            loss = loss_fn(pred, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 100 == 0:
                logging.info(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}')
                running_loss = 0.0

    out_model = f'model.{uuid.uuid4()}.pth'
    logging.info(f'saving model to {out_model}')
    torch.save(model.state_dict(), out_model)

    # since we're not training, we don't need to calculate the gradients for our outputs
    correct = 0
    total = 0
    with torch.no_grad():
        for (inputs, labels) in loader:
            pred = model(inputs)
            _, predicted = torch.max(pred.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Train accuracy: {100 * correct // total}%')

    # xbatch, ybatch = next(iter(loader))
    # fig = plt.figure()
    # for i, (x, y) in enumerate(zip(xbatch.numpy(), ybatch.numpy())):
    #     im = x.reshape(64, 60, 3)
    #     ax = fig.add_subplot(16, 16, i + 1)
    #     ax.imshow(im)
    #     ax.axis('off')
    # plt.show()


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(message)s',
                        datefmt='%Y%m%d %H:%M:%S',
                        level=logging.INFO,
                        stream=sys.stdout)
    main()

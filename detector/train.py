#!/usr/bin/env python
"""XXX
"""
import argparse
from datetime import datetime
import logging
import os
import sys
import uuid
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import cv2
import matplotlib.pyplot as plt
import numpy as np

from data import TileData
import ontology


class ConvNet(nn.Module):
    """XXX
    """
    def __init__(self):
        super().__init__()
        # XXX pad
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(2496, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 28)  # XXX 28 == num classes

    def forward(self, x):
        # XXX use F batch norm
        # XXX use F pool
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def evaluate(*, model, data, writer, label):
    """XXX
    """
    loader = DataLoader(data,
                        batch_size=64,
                        shuffle=False)

    predictions = []
    labels = []
    with torch.no_grad():
        for (x, y) in loader:
            ypred = model(x)
            _, pred = torch.max(ypred.data, 1)
            predictions.append(pred.numpy())
            labels.append(y.numpy())
    predictions = np.hstack(predictions)
    labels = np.hstack(labels)

    i_correct = np.where(predictions == labels)[0]
    i_incorrect = np.where(predictions != labels)[0]

    def _add_figure(index, label):
        np.random.shuffle(index)
        
        def _image(i):
            xi, yi = data[i]
            yi_pred = predictions[i]
            xi = xi.numpy().transpose((1, 2, 0))
            return cv2.putText(xi,
                               ontology.CLASSES_TO_LABELS[int(yi_pred)],
                               (5, 25), 
                               cv2.FONT_HERSHEY_SIMPLEX,
                               1.0,
                               (1, 0, 0))

        im = np.hstack([_image(i) for i in index[:25]])
        
        fig = plt.figure(figsize=(15, 2))
        ax = fig.add_subplot(111)
        ax.imshow(im)
        ax.axis('off')

        writer.add_figure(label, fig)

    _add_figure(i_correct, f'Good/{label}')
    _add_figure(i_incorrect, f'Bad/{label}')

    return (predictions == labels).sum() / len(labels)



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, help='path to config yaml')
    parser.add_argument('--model', help='path to model on disk')
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    logdir = os.path.join('runs', f'{datetime.now():%Y%m%d_%H%M%S}')
    writer = SummaryWriter(log_dir=logdir)

    train_data = TileData(dirs=config['train_data'],
                          fraction_other=config['fraction_other'])
    train_loader = DataLoader(train_data,
                              batch_size=config['batch_size'],
                              shuffle=True)
    logging.info(f'loaded {len(train_data)} train samples')

    model = ConvNet()
    if args.model:
        logging.info(f'loading model from {args.model}')
        model.load_state_dict(torch.load(args.model))

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

    for epoch in range(config['epochs']):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader, 0):
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            pred = model(inputs)
            loss = loss_fn(pred, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            writer.add_scalar('Loss/train',
                              loss.item(),
                              epoch * len(train_data) + i * len(labels))
        logging.info(f'[{epoch:4d}] loss: {running_loss:.3f}')

    logging.info('saving model...')
    torch.save(model.state_dict(), os.path.join(logdir, 'model.pth'))


    # Evaluate models.
    test_data = TileData(dirs=config['test_data'],
                         fraction_other=config['fraction_other'])
    logging.info(f'loaded {len(test_data)} test samples')

    train_accuracy = evaluate(model=model, data=train_data, writer=writer, label='Train')
    test_accuracy = evaluate(model=model, data=test_data, writer=writer, label='Test')
    logging.info(f'train accuracy {100 * train_accuracy:.3f}%')
    logging.info(f'test accuracy  {100 * test_accuracy:.3f}%')

    writer.close()


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(message)s',
                        datefmt='%Y%m%d %H:%M:%S',
                        level=logging.INFO,
                        stream=sys.stdout)
    main()

from __future__ import print_function, division

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import time
import copy
from models.baseline import BaselineMU
from data_helper import *
from torch.utils.data import DataLoader
import sys
plt.ion()   # interactive mode
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class BaselineSolver():
    def __init__(self, source_domain, target_domain, optimizer, criterion, num_epochs):
        self.model = None
        self.source_domain = source_domain
        self.target_domain = target_domain
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = None
        self.data_loader = None
        self.num_epochs = num_epochs

    def test_one_epoch(self, criterion):
        self.model.eval()

        total_loss = 0
        corrects = 0

        for inputs, labels in self.data_loader['target']['test']:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = self.model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * inputs.size(0)
            corrects += torch.sum(preds == labels.data)

        accuracy = corrects / len(self.data_loader['target']['test'])
        average_loss = total_loss / len(self.data_loader['target']['test'])

        return average_loss, accuracy

    def train_one_epoch(self, criterion):
        self.model.train()

        total_loss = 0
        corrects = 0

        data_num = len(self.data_loader['source']['train'].dataset)
        batch_size = self.data_loader['source']['train'].batch_size
        processed_num = 0

        batch_num = 0
        for inputs, labels in self.data_loader['source']['train']:
            sys.stdout.write('\r{}/{}'.format(processed_num,data_num))
            sys.stdout.flush()
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            self.optimizer.zero_grad()

            outputs = self.model(inputs)
            _, preds = torch.max(outputs, 1)

            loss = criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            corrects += torch.sum(preds == labels.data)
            processed_num += batch_size

        accuracy = corrects / data_num
        average_loss = total_loss/ data_num

        print()
        return average_loss, accuracy

    def train(self, num_epochs):
        since = time.time()

        best_model = copy.deepcopy(self.model.state_dict())
        best_test_accuracy = 0.0

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            # train on one epoch
            self.scheduler.step()
            train_loss, train_accuracy = self.train_one_epoch(criterion=self.criterion)
            print('Average Train Loss: {:.4f} Acc: {:.4f}'.format(train_loss, train_accuracy))

            # test on one epoch
            test_loss, test_accuracy = self.test_one_epoch(criterion=self.criterion)
            print('Average Test Loss: {:.4f} Acc: {:.4f}'.format(test_loss, train_accuracy))

            # save the best model
            if test_accuracy > best_test_accuracy:
                best_test_accuracy = test_accuracy
                best_model = copy.deepcopy(self.model.state_dict())

            print()

        time_elapsed = time.time() - since

        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best test Acc: {:4f}'.format(best_test_accuracy))

        # load best model weights
        self.model.load_state_dict(best_model)

    def solve(self):
        source_data = load_MNIST(root_dir='../data/Digits/MNIST', image_size=[28, 28], Gray_to_RGB=False)
        target_data = load_USPS(root_dir='../data/Digits/USPS', image_size=[28, 28], Gray_to_RGB=False)

        self.batch_size = 256

        source_data_loader = {
            'train': torch.utils.data.DataLoader(source_data['train'], batch_size=256, shuffle=True, num_workers=4),
            'test': torch.utils.data.DataLoader(source_data['test'], batch_size=256, shuffle=False, num_workers=4)
        }
        target_data_loader = {
            'train': torch.utils.data.DataLoader(target_data['train'], batch_size=256, shuffle=True, num_workers=4),
            'test': torch.utils.data.DataLoader(target_data['test'], batch_size=256, shuffle=False, num_workers=4)
        }

        self.data_loader = {
            'source': source_data_loader,
            'target': target_data_loader
        }

        self.model = BaselineMU(n_classes=10)
        self.model = self.model.to(device)

        if self.optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=7, gamma=0.1)

        self.train(
            num_epochs=self.num_epochs
        )

def main():
    solver = BaselineSolver(
        source_domain = 'MNIST',
        target_domain = 'USPS',
        optimizer = 'Adam',
        criterion = nn.CrossEntropyLoss(),
        num_epochs = 25
    )
    solver.solve()

if __name__ == '__main__':
    main()
from __future__ import print_function, division

import torch
import matplotlib.pyplot as plt
import time
from models.baseline import BaselineMU, BaselineStoM
from data_helper import *
import sys
import pandas as pd

plt.ion()  # interactive mode

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


class BaselineSolver():
    def __init__(self, source_domain, target_domain, optimizer, criterion, pre_trained, batch_size, num_epochs):
        self.model = None
        self.model_name = ''
        self.source_domain = source_domain
        self.target_domain = target_domain
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = None
        self.data_loader = None
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.pre_trained = pre_trained
        self.model_saving_path = ''
        self.model = None
        self.log_path = ''
        self.log = {
            'model': [],
            'source': [],
            'target': [],
            'epoch': [],
            'train_acc': [],
            'val_acc': [],
            'test_acc': [],
            'train_loss': [],
            'val_loss': [],
            'test_loss': []
        }

    def add_log(self, epoch, train_acc, val_acc, test_acc, train_loss, val_loss, test_loss):
        self.log['model'].append(self.model_name)
        self.log['source'].append(self.source_domain)
        self.log['target'].append(self.target_domain)
        self.log['epoch'].append(epoch)
        self.log['train_acc'].append(train_acc)
        self.log['val_acc'].append(val_acc)
        self.log['test_acc'].append(test_acc)
        self.log['train_loss'].append(train_loss)
        self.log['val_loss'].append(val_loss)
        self.log['test_loss'].append(test_loss)

    def test(self, data_loader, criterion):
        self.model.eval()

        total_loss = 0
        corrects = 0
        data_num = len(data_loader.dataset)

        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = self.model(inputs)
            _, preds = torch.max(outputs, 1)

            loss = criterion(outputs, labels)

            total_loss += loss.item() * inputs.size(0)
            corrects += (preds == labels.data).sum().item()

        acc = corrects / data_num
        average_loss = total_loss / data_num
        print('Datasize = {} , corrects = {}'.format(data_num, corrects))

        return average_loss, acc

    def train_one_epoch(self, data_loader, criterion):
        self.model.train()

        total_loss = 0
        corrects = 0

        data_num = len(data_loader.dataset)
        batch_size = data_loader.batch_size
        processed_num = 0

        batch_num = 0
        for inputs, labels in data_loader:
            sys.stdout.write('\r{}/{}'.format(processed_num, data_num))
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
            corrects += (preds == labels.data).sum().item()
            processed_num += batch_size

        acc = corrects / data_num
        average_loss = total_loss / data_num
        print()
        print('Datasize = {} , corrects = {}'.format(data_num, corrects))

        return average_loss, acc

    def train(self, num_epochs):
        since = time.time()

        best_val_loss, best_val_acc = self.test(
            data_loader=self.data_loader['val'],
            criterion=self.criterion
        )
        print('Initial Val Loss: {:.4f} Acc: {:.4f}\n'.format(best_val_loss, best_val_acc))
        print()

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            # self.scheduler.step()
            train_loss, train_acc = self.train_one_epoch(
                data_loader=self.data_loader['train'],
                criterion=self.criterion
            )
            print('Train Loss: {:.4f} Acc: {:.4f}\n'.format(train_loss, train_acc))

            val_loss, val_acc = self.test(
                data_loader=self.data_loader['val'],
                criterion=self.criterion
            )
            print('Val Loss: {:.4f} Acc: {:.4f}\n'.format(val_loss, val_acc))

            test_loss, test_acc = self.test(
                data_loader=self.data_loader['test'],
                criterion=self.criterion
            )
            print('Test Loss: {:.4f} Acc: {:.4f}\n'.format(test_loss, test_acc))

            self.add_log(epoch, train_acc, val_acc, test_acc, train_loss, val_loss, test_loss)

            pd.DataFrame(
                data=self.log,
                columns=['model', 'source', 'target', 'epoch', 'train_acc', 'val_acc', 'test_acc', 'train_loss',
                         'val_loss', 'test_loss']
            ).to_csv(self.log_path, index=0)

            # save the best model
            if val_acc >= best_val_acc:
                best_val_acc = val_acc
                best_val_loss = val_loss
                self.save_model(model=self.model, path=self.model_saving_path)

            print()

        time_elapsed = time.time() - since

        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best test Acc: {:4f}\n'.format(best_val_acc))

    def save_model(self, model, path):
        print('New model is better, start saving ......')
        torch.save(model.state_dict(), path)
        print('Save model in {} successfully\n'.format(path))

    def load_model(self, model, path):
        if os.path.exists(path):
            model.load_state_dict(torch.load(path))
            print('Read model in {} successfully\n'.format(path))
        else:
            print('Cannot find {}, use the initial model\n'.format(path))

    def solve(self):
        source_data = None
        target_data = None

        if self.source_domain == 'MNIST' and self.target_domain == 'USPS':
            source_data = load_MNIST(root_dir='./data/Digits/MNIST', image_size=[28, 28], Gray_to_RGB=False)
            target_data = load_USPS(root_dir='./data/Digits/USPS', image_size=[28, 28], Gray_to_RGB=False)
            self.model = BaselineMU(n_classes=10)
            self.model_saving_path = './models_checkpoints/BaselineMtoU.pt'
            self.log_path = './logs/BaselineMtoU.csv'
            self.model_name = 'BaselineMtoU'

        if self.source_domain == 'USPS' and self.target_domain == 'MNIST':
            source_data = load_USPS(root_dir='./data/Digits/USPS', image_size=[28, 28], Gray_to_RGB=False)
            target_data = load_MNIST(root_dir='./data/Digits/MNIST', image_size=[28, 28], Gray_to_RGB=False)
            self.model = BaselineMU(n_classes=10)
            self.model_saving_path = './models_checkpoints/BaselineUtoM.pt'
            self.log_path = './logs/BaselineUtoM.csv'
            self.model_name = 'BaselineUtoM'

        if self.source_domain == 'SVHN' and self.target_domain == 'MNIST':
            source_data = load_SVHN(root_dir='./data/Digits/SVHN', image_size=[32, 32])
            target_data = load_MNIST(root_dir='./data/Digits/MNIST', image_size=[32, 32], Gray_to_RGB=True)
            self.model = BaselineStoM(n_classes=10)
            self.model_saving_path = './models_checkpoints/BaselineStoM.pt'
            self.log_path = './logs/BaselineStoM.csv'
            self.model_name = 'BaselineStoM'

        print('Source domain :{}, Data size:{}'.format(self.source_domain, len(source_data['train'])))
        print('Target domain :{}, Data size:{}'.format(self.target_domain, len(target_data['test'])))

        self.data_loader = {
            'train': torch.utils.data.DataLoader(source_data['train'], batch_size=self.batch_size, shuffle=True,
                                                 num_workers=8),
            'val': torch.utils.data.DataLoader(source_data['test'], batch_size=round(self.batch_size * 1.5),
                                               shuffle=True, num_workers=8),
            'test': torch.utils.data.DataLoader(target_data['test'], batch_size=round(self.batch_size * 1.5),
                                                shuffle=False, num_workers=8),
        }

        self.model = self.model.to(device)

        if self.pre_trained:
            self.load_model(self.model, self.model_saving_path)

        if self.optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        self.train(
            num_epochs=self.num_epochs
        )


solverMtoU = BaselineSolver(
    source_domain = 'MNIST',
    target_domain = 'USPS',
    optimizer = 'Adam',
    criterion = nn.CrossEntropyLoss(),
    batch_size = 256,
    num_epochs = 30,
    pre_trained = False,
)

solverUtoM = BaselineSolver(
    source_domain = 'USPS',
    target_domain = 'MNIST',
    optimizer = 'Adam',
    criterion = nn.CrossEntropyLoss(),
    batch_size = 256,
    num_epochs = 30,
    pre_trained = False,
)

solverStoM = BaselineSolver(
    source_domain = 'SVHN',
    target_domain = 'MNIST',
    optimizer = 'Adam',
    criterion = nn.CrossEntropyLoss(),
    batch_size =256,
    num_epochs = 30,
    pre_trained = False,
)

solverAtoW = BaselineSolver(
    source_domain = 'Amazon',
    target_domain = 'Webcam',
    optimizer = 'Adam',
    criterion = nn.CrossEntropyLoss(),
    batch_size =256,
    num_epochs = 30,
    pre_trained = False,
)

solverStoM = BaselineSolver(
    source_domain = 'Dslr',
    target_domain = 'Webcam',
    optimizer = 'Adam',
    criterion = nn.CrossEntropyLoss(),
    batch_size =256,
    num_epochs = 30,
    pre_trained = False,
)

solverStoM = BaselineSolver(
    source_domain = 'Webcam',
    target_domain = 'Dslr',
    optimizer = 'Adam',
    criterion = nn.CrossEntropyLoss(),
    batch_size =256,
    num_epochs = 30,
    pre_trained = False,
)
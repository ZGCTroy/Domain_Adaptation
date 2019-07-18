from __future__ import print_function, division
import torch
import torch.nn as nn
import torchvision
import time
from models.baseline import BaselineMU, BaselineStoM
from data_helper import *
import sys
import pandas as pd


class BaselineSolver():
    def __init__(self, dataset_type, source_domain, target_domain, cuda , optimizer = 'Adam', criterion = nn.CrossEntropyLoss(), pretrained = False, batch_size = 32,
                 num_epochs = 200, if_test=True, test_mode=False, num_workers=4):
        self.model = None
        self.model_name = ''
        self.dataset_type = dataset_type
        self.source_domain = source_domain
        self.target_domain = target_domain
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = None
        self.data_loader = {
            'train': None,
            'val': None,
            'test': None
        }
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.pretrained = pretrained
        self.model_saving_path = {'train': '', 'test': ''}
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
        self.num_workers = num_workers
        self.if_test = if_test
        self.test_mode = test_mode
        self.cuda = cuda
        self.device = torch.device(self.cuda if torch.cuda.is_available() else "cpu")

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
        batch_size = data_loader.batch_size
        processed_num = 0

        for inputs, labels in data_loader:
            sys.stdout.write('\r{}/{}'.format(processed_num, data_num))
            sys.stdout.flush()
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(inputs)
            _, preds = torch.max(outputs, 1)

            loss = criterion(outputs, labels)

            total_loss += loss.item() * inputs.size(0)
            corrects += (preds == labels.data).sum().item()
            processed_num += batch_size

        acc = corrects / data_num
        average_loss = total_loss / data_num
        print('\nDatasize = {} , corrects = {}'.format(data_num, corrects))

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
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

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
        print('\nDatasize = {} , corrects = {}'.format(data_num, corrects))

        return average_loss, acc

    def train_Office(self, num_epochs):
        since = time.time()

        best_train_loss, best_train_acc = self.test(
            data_loader=self.data_loader['train'],
            criterion=self.criterion
        )
        print('Initial Train Loss: {:.4f} Acc: {:.4f}\n'.format(best_train_loss, best_train_acc))
        print()

        if self.if_test:
            best_test_loss, best_test_acc = self.test(
                data_loader=self.data_loader['test'],
                criterion=self.criterion
            )
            print('Initial Test Loss: {:.4f} Acc: {:.4f}\n'.format(best_test_loss, best_test_acc))
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

            if self.if_test:
                test_loss, test_acc = self.test(
                    data_loader=self.data_loader['test'],
                    criterion=self.criterion
                )
                print('Test Loss: {:.4f} Acc: {:.4f}\n'.format(test_loss, test_acc))
                if test_acc >= best_test_acc:
                    best_test_acc = test_acc
                    best_test_loss = test_loss
                    self.save_model(model=self.model, path=self.model_saving_path['test'])
            else:
                test_loss = test_acc = 0

            if train_acc >= best_train_acc:
                best_train_acc = train_acc
                best_train_loss = train_loss
                self.save_model(model=self.model, path=self.model_saving_path['train'])

            self.add_log(epoch, train_acc, 0, test_acc, train_loss, 0, test_loss)

            pd.DataFrame(
                data=self.log,
                columns=['model', 'source', 'target', 'epoch', 'train_acc', 'val_acc', 'test_acc', 'train_loss',
                         'val_loss', 'test_loss']
            ).to_csv(self.log_path, index=0)

            print()

        time_elapsed = time.time() - since

        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best Train Acc: {:4f}\n'.format(best_train_acc))

        if self.if_test:
            print('Best Test Acc: {:4f}\n'.format(best_test_acc))

    def train_Digits(self, num_epochs):
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

            if self.if_test:
                test_loss, test_acc = self.test(
                    data_loader=self.data_loader['test'],
                    criterion=self.criterion
                )
                print('Test Loss: {:.4f} Acc: {:.4f}\n'.format(test_loss, test_acc))
            else:
                test_loss = test_acc = 0

            if val_acc >= best_val_acc:
                best_val_acc = val_acc
                best_val_loss = val_loss
                self.save_model(model=self.model, path=self.model_saving_path['train'])

            self.add_log(epoch, train_acc, val_acc, test_acc, train_loss, val_loss, test_loss)

            pd.DataFrame(
                data=self.log,
                columns=['model', 'source', 'target', 'epoch', 'train_acc', 'val_acc', 'test_acc', 'train_loss',
                         'val_loss', 'test_loss']
            ).to_csv(self.log_path, index=0)

            print()

        time_elapsed = time.time() - since

        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best Val Acc: {:4f}\n'.format(best_val_acc))

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
            source_data = load_MNIST(root_dir='./data/Digits/MNIST', resize_size=28, Gray_to_RGB=False)
            target_data = load_USPS(root_dir='./data/Digits/USPS', resize_size=28, Gray_to_RGB=False)
            self.model = BaselineMU(n_classes=10)
            self.model_name = 'BaselineMtoU'

        if self.source_domain == 'USPS' and self.target_domain == 'MNIST':
            source_data = load_USPS(root_dir='./data/Digits/USPS', resize_size=28, Gray_to_RGB=False)
            target_data = load_MNIST(root_dir='./data/Digits/MNIST', resize_size=28, Gray_to_RGB=False)
            self.model = BaselineMU(n_classes=10)
            self.model_name = 'BaselineUtoM'

        if self.source_domain == 'SVHN' and self.target_domain == 'MNIST':
            source_data = load_SVHN(root_dir='./data/Digits/SVHN', resize_size=32)
            target_data = load_MNIST(root_dir='./data/Digits/MNIST', resize_size=32, Gray_to_RGB=True)
            self.model = BaselineStoM(n_classes=10)
            self.model_name = 'BaselineStoM'

        if self.source_domain == 'Amazon' and self.target_domain == 'Webcam':
            source_data = load_Amazon(root_dir='./data/Office31/Amazon', resize_size=256, crop_size=224)
            target_data = load_Webcam(root_dir='./data/Office31/Webcam', resize_size=256, crop_size=224)
            self.model = torchvision.models.resnet50(pretrained=True)
            self.model.fc = nn.Linear(self.model.fc.in_features, 31)
            self.model_name = 'Resnet50_AtoW'

        if self.source_domain == 'Dslr' and self.target_domain == 'Webcam':
            source_data = load_Dslr(root_dir='./data/Office31/Dslr', resize_size=256, crop_size=224)
            target_data = load_Webcam(root_dir='./data/Office31/Webcam', resize_size=256, crop_size=224)
            self.model = torchvision.models.resnet50(pretrained=True)
            self.model.fc = nn.Linear(self.model.fc.in_features, 31)
            self.model_name = 'Resnet50_DtoW'

        if self.source_domain == 'Webcam' and self.target_domain == 'Dslr':
            source_data = load_Webcam(root_dir='./data/Office31/Webcam', resize_size=256, crop_size=224)
            target_data = load_Dslr(root_dir='./data/Office31/Dslr', resize_size=256, crop_size=224)
            self.model = torchvision.models.resnet50(pretrained=True)
            self.model.fc = nn.Linear(self.model.fc.in_features, 31)
            self.model_name = 'Resnet50_WtoD'

        print(
            'Source domain :{}, Data size:{}'.format(self.source_domain, len(source_data['train'])))
        print(
            'Target domain :{}, Data size:{}'.format(self.target_domain, len(target_data['test'])))

        self.log_path = os.path.join('./logs', self.model_name + '.csv')
        self.model_saving_path['train'] = os.path.join('./models_checkpoints', self.model_name + '_best_train.pt')

        if self.pretrained:
            self.load_model(self.model, self.model_saving_path['train'])

        if self.test_mode:
            self.test(
                data_loader=torch.utils.data.DataLoader(
                    target_data['test'],
                    batch_size=self.batch_size,
                    shuffle=False, num_workers=self.num_workers
                ),
                criterion=self.criterion
            )
            return

        if self.optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        self.data_loader['train'] = torch.utils.data.DataLoader(
            source_data['train'],
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )

        if self.if_test:
            self.data_loader['test'] = torch.utils.data.DataLoader(
                target_data['test'],
                batch_size=self.batch_size,
                shuffle=False, num_workers=self.num_workers
            )
            self.model_saving_path['test'] = os.path.join('./models_checkpoints', self.model_name + '_best_test.pt')

        self.model = self.model.to(self.device)

        if self.dataset_type == 'Digits':
            self.data_loader['val'] = torch.utils.data.DataLoader(
                source_data['test'],
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers
            )
            self.train_Digits(num_epochs=self.num_epochs)
        else:
            self.train_Office(num_epochs=self.num_epochs)



from __future__ import print_function, division
import torch
import torch.nn as nn
from data_helper import *
import pandas as pd
import time
from torch.utils.data import DataLoader


class Solver():
    def __init__(self, dataset_type, source_domain, target_domain, cuda='cuda:0',
                 pretrained=False, batch_size=32,
                 num_epochs=999999, max_iter_num=999999, test_interval=500, test_mode=False, num_workers=2,
                 clean_log=False):
        self.dataset_type = dataset_type
        self.source_domain = source_domain
        self.target_domain = target_domain
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.pretrained = pretrained
        self.num_workers = num_workers
        self.test_interval = test_interval
        self.max_iter_num = max_iter_num
        self.test_mode = test_mode
        self.cuda = cuda
        self.device = torch.device(self.cuda if torch.cuda.is_available() else "cpu")
        self.clean_log = clean_log
        self.model = None
        self.model_name = None
        self.scheduler = None
        self.data_loader = {
            'source': {
                'train': None,
                'test': None
            },
            'target': {
                'train': None,
                'test': None
            }
        }

        self.log = {
            'iter': [],
            'epoch': [],
            'source': [],
            'target': [],
            'model': [],
            'optimizer': [],
            'batch_size': [],
            'lr': [],
            'train_acc': [],
            'val_acc': [],
            'test_acc': [],
            'train_loss': [],
            'val_loss': [],
            'test_loss': []
        }

        self.optimizer = None
        self.source_data = {}
        self.target_data = {}
        self.n_classes = 0
        self.task = ''
        self.logs_dir = ''
        self.models_checkpoints_dir = ''
        self.iter_num = 0
        self.lr = 0.01
        self.optimizer_type = 'SGD'


    def cycle(self, iterable):
        while True:
            for x in iterable:
                yield x

    def test(self, data_loader):
        raise NotImplementedError

    def train_one_epoch(self):
        raise NotImplementedError

    def train(self, num_epochs):
        since = time.time()

        self.iter_num = 0
        log_iter = 0

        best_val_loss, best_val_acc = self.test(
            data_loader=self.data_loader['source']['test'],
        )

        print('Initial Train Loss: {:.4f} Acc: {:.4f}\n'.format(best_val_loss, best_val_acc))
        print()

        best_test_loss, best_test_acc = self.test(
            data_loader=self.data_loader['target']['test'],
        )
        print('Initial Test Loss: {:.4f} Acc: {:.4f}\n'.format(best_val_loss, best_val_acc))
        print()

        for epoch in range(num_epochs):
            self.epoch = epoch
            print('\nEpoch {}/{}'.format(epoch, num_epochs - 1), '\n', '-' * 10)
            print('iteration : {}\n'.format(self.iter_num))

            # TODO 1 : Train
            train_loss, train_acc = self.train_one_epoch()

            print('Train Loss: {:.4f} Acc: {:.4f}\n'.format(train_loss, train_acc))

            # TODO 2 : Validation
            val_acc = val_loss = 0
            if self.dataset_type == 'Digits':

                val_loss, val_acc = self.test(data_loader=self.data_loader['source']['test'], )
                print('Val Loss: {:.4f} Acc: {:.4f}\n'.format(val_loss, val_acc))

                if val_acc >= best_val_acc:
                    best_val_acc = val_acc
                    best_val_loss = val_loss
                    self.save_model(path=self.models_checkpoints_dir + '/' + self.model_name + '_best_train.pt')

            # TODO 3 : Test
            if self.iter_num - log_iter >= self.test_interval:
                log_iter = self.iter_num
                test_loss, test_acc = self.test(data_loader=self.data_loader['target']['test'])

                print('Test Loss: {:.4f} Acc: {:.4f}\n'.format(test_loss, test_acc))

                if test_acc >= best_test_acc:
                    best_test_acc = test_acc
                    best_test_loss = test_loss
                    self.save_model(path=self.models_checkpoints_dir + '/' + self.model_name + '_best_test.pt')

                self.add_log(epoch, train_acc, val_acc, test_acc, train_loss, val_loss, test_loss)
                self.save_log()

            if self.iter_num >= self.max_iter_num:
                break

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best Val Acc : {:4f}, Test Acc : {:4f}\n'.format(best_val_acc, best_test_acc))

    def set_model(self):
        raise NotImplementedError

    def set_optimizer(self):
        self.optimizer_type = 'SGD'

        self.lr = 0.001

        if self.task in ['AtoD', 'DtoW']:
            self.lr = 0.0003

        if self.task in ['StoM']:
            self.lr = 0.03

        if self.task in ['MtoU', 'UtoM']:
            self.lr = 0.02

        if self.dataset_type == 'OfficeHome':
            self.lr = 0.001

        self.optimizer = torch.optim.SGD(
            self.model.get_parameters(),
            lr=self.lr,
            momentum=0.9,
            weight_decay=0.0005,
            nesterov=True
        )

    def set_dataloader(self):
        self.data_loader['source']['train'] = DataLoader(
            self.source_data['train'],
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )
        self.data_loader['source']['test'] = DataLoader(
            self.source_data['test'],
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )

        self.data_loader['target']['train'] = DataLoader(
            self.target_data['train'],
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,

        )

        self.data_loader['target']['test'] = DataLoader(
            self.target_data['test'],
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def update_optimizer(self, gamma=10, power=0.75, weight_decay=0.0005):
        """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
        if self.num_epochs != 999999:
            p = self.epoch / self.num_epochs
        else:
            p = self.iter_num / self.max_iter_num

        lr = self.lr * (1.0 + gamma * p) ** (-power)

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr * param_group['lr_mult']
            param_group['weight_decay'] = weight_decay * param_group['decay_mult']

    def load_dataset(self):
        # TODO 1 : Load Dataset
        if self.dataset_type == 'Digits':
            self.n_classes = 10
            self.task = self.source_domain[0] + 'to' + self.target_domain[0]
            if self.task == 'MtoU':
                self.source_data = load_MNIST(root_dir='./data/Digits/MNIST')
                self.target_data = load_USPS(root_dir='./data/Digits/USPS')

            if self.task == 'UtoM':
                self.source_data = load_USPS(root_dir='./data/Digits/USPS')
                self.target_data = load_MNIST(root_dir='./data/Digits/MNIST')

            if self.task == 'StoM':
                self.source_data = load_SVHN(root_dir='./data/Digits/SVHN')
                self.target_data = load_MNIST(root_dir='./data/Digits/MNIST', resize_size=32, Gray_to_RGB=True)

        if self.dataset_type == 'Office31':
            self.n_classes = 31
            self.task = self.source_domain[0] + 'to' + self.target_domain[0]
            self.source_data = load_Office('./data/Office31', domain=self.source_domain)
            self.target_data = load_Office('./data/Office31', domain=self.target_domain)

        if self.dataset_type == 'OfficeHome':
            self.n_classes = 65
            self.task = self.source_domain[:2] + 'to' + self.target_domain[:2]
            self.source_data = load_Office('./data/OfficeHome', domain=self.source_domain)
            self.target_data = load_Office('./data/OfficeHome', domain=self.target_domain)

        print('Source train domain :{}, Data size:{}'.format(self.source_domain, len(self.source_data['train'])))
        print('Target test domain :{}, Data size:{}'.format(self.target_domain, len(self.target_data['test'])))

    def solve(self):
        # TODO 1 : load dataset
        self.load_dataset()

        # TODO 2 : set dataloader
        self.set_dataloader()

        # TODO 3 : set model
        self.set_model()

        # TODO 4 : set optimizer
        self.set_optimizer()

        # TODO 5 : set other parameters
        self.models_checkpoints_dir = './models_checkpoints/' + self.dataset_type + '/' + self.task
        self.logs_dir = './logs/' + self.dataset_type + '/' + self.task

        if self.test_mode:
            self.test(
                data_loader=torch.utils.data.DataLoader(
                    self.target_data['test'],
                    batch_size=self.batch_size,
                    shuffle=False, num_workers=self.num_workers
                )
            )
        else:
            self.train(num_epochs=self.num_epochs)

    def add_log(self, epoch, train_acc, val_acc, test_acc, train_loss, val_loss, test_loss):
        self.log['iter'].append(self.iter_num)
        self.log['epoch'].append(epoch)
        self.log['source'].append(self.source_domain)
        self.log['target'].append(self.target_domain)
        self.log['model'].append(self.model_name)
        self.log['optimizer'].append(self.optimizer_type)
        self.log['batch_size'].append(self.batch_size)
        self.log['lr'].append(self.lr)
        self.log['train_acc'].append('%.4f' % train_acc)
        self.log['val_acc'].append('%.4f' % val_acc)
        self.log['test_acc'].append('%.4f' % test_acc)
        self.log['train_loss'].append('%.4f' % train_loss)
        self.log['val_loss'].append('%.4f' % val_loss)
        self.log['test_loss'].append('%.4f' % test_loss)

    def save_log(self):
        path = os.path.join(self.logs_dir, self.model_name + '.csv')

        log = pd.DataFrame(
            data=self.log,
            columns=['iter', 'epoch', 'source', 'target', 'model', 'optimizer', 'batch_size', 'lr', 'train_acc',
                     'val_acc',
                     'test_acc',
                     'train_loss', 'val_loss', 'test_loss']
        )

        log.to_csv(path, mode='w', index=0)

        print('successfully save log in {}'.format(path))

    def save_model(self, path):
        print('New model is better, start saving ......')
        torch.save(self.model.state_dict(), path)
        print('Save model in {} successfully\n'.format(path))

    def load_model(self, path):
        if os.path.exists(path):
            self.model.load_state_dict(torch.load(path))
            print('Read model in {} successfully\n'.format(path))
        else:
            print('Cannot find {}, use the initial model\n'.format(path))

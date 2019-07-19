from __future__ import print_function, division
import torch
import torch.nn as nn
from data_helper import *
import pandas as pd

class Solver():
    def __init__(self, source_domain, target_domain, cuda, optimizer='Adam', criterion=nn.CrossEntropyLoss(),
                 pretrained=False, batch_size=32,
                 num_epochs=300, if_test=True, test_mode=False, num_workers=2):
        self.model = None
        self.model_name = ''
        self.source_domain = source_domain
        self.target_domain = target_domain
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = None
        self.data_loader = {'train': None, 'val': None, 'test': None}
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
        self.source_data = {}
        self.target_data = {}

    def load_dataset_and_set_model(self):
        raise NotImplementedError

    def test(self, data_loader, criterion):
        raise NotImplementedError

    def train_one_epoch(self, data_loader, criterion):
        raise NotImplementedError

    def train(self, num_epochs):
        raise NotImplementedError

    def solve(self):
        self.load_dataset_and_set_model()

        self.log_path = os.path.join('./logs', self.model_name + '.csv')
        self.model_saving_path['train'] = os.path.join('./models_checkpoints', self.model_name + '_best_train.pt')

        if self.test_mode:
            self.test(
                data_loader=torch.utils.data.DataLoader(
                    self.target_data['test'],
                    batch_size=self.batch_size,
                    shuffle=False, num_workers=self.num_workers
                ),
                criterion=self.criterion
            )
            return

        if self.optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        self.data_loader['train'] = torch.utils.data.DataLoader(
            self.source_data['train'],
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )

        if self.if_test:
            self.data_loader['test'] = torch.utils.data.DataLoader(
                self.target_data['test'],
                batch_size=self.batch_size,
                shuffle=False, num_workers=self.num_workers
            )
            self.model_saving_path['test'] = os.path.join('./models_checkpoints', self.model_name + '_best_test.pt')

        self.train(num_epochs=self.num_epochs)

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

    def save_log(self):
        pd.DataFrame(
            data=self.log,
            columns=['model', 'source', 'target', 'epoch', 'train_acc', 'val_acc', 'test_acc', 'train_loss',
                     'val_loss', 'test_loss']
        ).to_csv(self.log_path, index=0)

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
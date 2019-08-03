from __future__ import print_function, division

import sys
import time
import torch
import torch.nn as nn

from networks.Baseline import DigitsStoM, DigitsMU, ResNet50
from solvers.Solver import Solver


class BaselineSolver(Solver):

    def __init__(self, dataset_type, source_domain, target_domain, cuda, pretrained=False,
                 batch_size=32,
                 num_epochs=99999, max_iter_num=99999999, test_interval=100, test_mode=False, num_workers=2, lr=0.001,
                 gamma=10, optimizer_type='SGD'):
        super(BaselineSolver, self).__init__(
            dataset_type=dataset_type,
            source_domain=source_domain,
            target_domain=target_domain,
            cuda=cuda,
            pretrained=pretrained,
            batch_size=batch_size,
            num_epochs=num_epochs,
            max_iter_num=max_iter_num,
            test_interval=test_interval,
            test_mode=test_mode,
            num_workers=num_workers,
            lr=lr,
            gamma=gamma,
            optimizer_type=optimizer_type
        )
        self.model_name = 'Baseline'

    def set_model(self):
        if self.dataset_type == 'Digits':
            if self.task == 'StoM':
                self.model = DigitsStoM(n_classes=self.n_classes)
            if self.task in ['MtoU', 'UtoM']:
                self.model = DigitsMU(n_classes=self.n_classes)

        if self.dataset_type in ['Office31', 'OfficeHome']:
            self.model = ResNet50(bottleneck_dim=256, n_classes=self.n_classes, pretrained=True)

        if self.pretrained:
            self.load_model(path=self.models_checkpoints_dir + '/' + self.model_name + '_best_train.pt')

        self.model = self.model.to(self.device)

    def test(self, data_loader):
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

            class_outputs = self.model(inputs, get_features=False, get_class_outputs=True)

            _, preds = torch.max(class_outputs, 1)

            corrects += (preds == labels.data).sum().item()
            processed_num += batch_size

        acc = corrects / data_num
        average_loss = total_loss / data_num

        print('\nDatasize = {} , corrects = {}'.format(data_num, corrects))

        return average_loss, acc

    def train_one_epoch(self):
        since = time.time()
        self.model.train()

        total_loss = 0
        corrects = 0

        data_num = len(self.data_loader['source']['train'].dataset)
        processed_num = 0

        criterion = nn.CrossEntropyLoss()
        for inputs, labels in self.data_loader['source']['train']:
            sys.stdout.write('\r{}/{}'.format(processed_num, data_num))
            sys.stdout.flush()

            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            self.update_optimizer()

            self.optimizer.zero_grad()

            class_outputs = self.model(inputs, get_features=False, get_class_outputs=True)

            _, preds = torch.max(class_outputs, 1)

            loss = criterion(class_outputs, labels)
            loss.backward()

            self.optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            corrects += (preds == labels.data).sum().item()
            processed_num += self.batch_size
            self.iter_num += 1

        acc = corrects / data_num
        average_loss = total_loss / data_num

        print()
        print('\nData size = {} , corrects = {}'.format(data_num, corrects))
        print('Using {:4f}'.format(time.time() - since))

        return average_loss, acc

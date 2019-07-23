from __future__ import print_function, division
import torch.nn as nn
import time
from data_helper import *
import sys
from solvers.Solver import Solver
from lr_scheduler.inv_lr_scheduler import inv_lr_scheduler
from network import DANN


class DANNSolver(Solver):

    def __init__(self, dataset_type, source_domain, target_domain, cuda='cuda:0',
                 pretrained=False,
                 batch_size=32,
                 num_epochs=9999, max_iter_num=9999999, test_interval=500, test_mode=False, num_workers=2,
                 clean_log=False):
        super(DANNSolver, self).__init__(
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
            clean_log=clean_log,
        )
        self.model_name = 'DANN'
        self.iter_num = 0

    def set_model(self):
        if self.dataset_type == 'Digits':
            if self.task in ['MtoU', 'UtoM']:
                self.model = DANN(n_classes=self.n_classes, base_model='DigitsMU')
            if self.task in ['StoM']:
                self.model = DANN(n_classes=self.n_classes, base_model='DigitsStoM')

        if self.dataset_type in ['Office31', 'OfficeHome']:
            self.model = DANN(n_classes=self.n_classes, base_model='ResNet50')

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

            # print('inputs ',inputs.size())
            # print('labels ',labels.size())

            class_outputs = self.model(inputs, test_mode=True)

            # print('class outputs ',class_outputs.size())

            _, preds = torch.max(class_outputs, 1)
            # print('preds ',preds.size())

            loss = nn.CrossEntropyLoss()(class_outputs, labels)

            total_loss += loss.item() * inputs.size(0)
            corrects += (preds == labels.data).sum().item()
            processed_num += batch_size

        acc = corrects / data_num
        average_loss = total_loss / data_num
        print('\nData size = {} , corrects = {}'.format(data_num, corrects))

        return average_loss, acc

    def train_one_epoch(self):
        since = time.time()
        self.model.train()

        total_loss = 0
        source_corrects = 0

        total_target_num = len(self.data_loader['target']['train'].dataset)
        processed_target_num = 0
        total_source_num = 0

        source_iter = iter(self.cycle(self.data_loader['source']['train']))

        for target_inputs, target_labels in self.data_loader['target']['train']:
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            sys.stdout.write('\r{}/{}'.format(processed_target_num, total_target_num))
            sys.stdout.flush()

            self.optimizer = inv_lr_scheduler(optimizer=self.optimizer, iter_num=self.iter_num, lr=self.lr)

            self.optimizer.zero_grad()

            # TODO 1 : Target Train

            target_inputs = target_inputs.to(self.device)

            target_domain_outputs = self.model(target_inputs, alpha=alpha, test_mode=False, is_source=False)

            # TODO 2 : Source Train

            source_inputs, source_labels = next(source_iter)
            source_inputs = source_inputs.to(self.device)
            source_labels = source_labels.to(self.device)

            source_domain_outputs, source_class_outputs = self.model(source_inputs, alpha=alpha, test_mode=False,
                                                                     is_source=True)

            # TODO 3 : LOSS

            target_domain_labels = torch.ones((target_labels.size()[0], 1), device=self.device)
            source_domain_labels = torch.zeros((source_labels.size()[0], 1), device=self.device)

            # print(target_domain_labels.size())
            # print(target_domain_outputs.size())

            source_class_loss = nn.CrossEntropyLoss()(source_class_outputs, source_labels)
            source_domain_loss = nn.BCELoss()(source_domain_outputs, source_domain_labels)
            target_domain_loss = nn.BCELoss()(target_domain_outputs, target_domain_labels)

            loss = target_domain_loss + source_domain_loss + source_class_loss
            loss.backward()

            self.optimizer.step()

            # TODO 5 : other parameters
            total_loss += loss.item() * source_labels.size()[0]
            _, source_class_preds = torch.max(source_class_outputs, 1)
            source_corrects += (source_class_preds == source_labels.data).sum().item()
            total_source_num += source_labels.size()[0]
            processed_target_num += target_labels.size()[0]
            self.iter_num += 1

        acc = source_corrects / total_source_num
        average_loss = total_loss / total_target_num

        print()
        print('\nData size = {} , corrects = {}'.format(total_source_num, source_corrects))
        print('Using {:4f}'.format(time.time() - since))
        return average_loss, acc

from __future__ import print_function, division
import torch.nn as nn
import time
from data_helper import *
import sys
from solvers.Solver import Solver
from network import MCD2
import numpy as np
from torch.nn import functional as F


class MCD2Solver(Solver):

    def __init__(self, dataset_type, source_domain, target_domain, cuda='cuda:0',
                 pretrained=False,
                 batch_size=36,
                 num_epochs=9999, max_iter_num=9999999, test_interval=500, test_mode=False, num_workers=2,
                 clean_log=False, lr=0.001, gamma=10, loss_weight=3.0, optimizer_type='SGD', num_k=4):
        super(MCD2Solver, self).__init__(
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
            lr=lr,
            gamma=gamma,
            optimizer_type=optimizer_type
        )
        self.model_name = 'MCD2'
        self.iter_num = 0
        self.optimizer_generator = None
        self.optimizer_classifier1 = None
        self.optimizer_classifier2 = None
        self.optimizer_decision_layer = None
        self.num_k = num_k

    def set_model(self):
        if self.dataset_type == 'Digits':
            if self.task in ['MtoU', 'UtoM']:
                self.model = MCD2(n_classes=self.n_classes, base_model='DigitsMU')
            if self.task in ['StoM']:
                self.model = MCD2(n_classes=self.n_classes, base_model='DigitsStoM')

        if self.dataset_type in ['Office31', 'OfficeHome']:
            self.model = MCD2(n_classes=self.n_classes, base_model='ResNet50')

        if self.pretrained:
            self.load_model(path=self.models_checkpoints_dir + '/' + self.model_name + '_best_train.pt')

        self.model = self.model.to(self.device)

    def test(self, data_loader):
        self.model.eval()

        corrects = 0
        data_num = len(data_loader.dataset)
        processed_num = 0

        for inputs, labels in data_loader:
            sys.stdout.write('\r{}/{}'.format(processed_num, data_num))
            sys.stdout.flush()

            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            outputs1, outputs2, decisions = self.model(inputs)

            outputs = nn.Softmax(dim=1)(outputs1) + nn.Softmax(dim=1)(outputs2)

            _, preds = torch.max(outputs, 1)

            corrects += (preds == labels.data).sum().item()

            processed_num += labels.size()[0]

        acc = corrects / processed_num
        print('\nData size = {} , corrects = {}'.format(processed_num, corrects))

        return 0, acc

    def set_optimizer(self):
        if self.optimizer_type == 'Adam':
            self.lr = 0.001

            self.optimizer_generator = torch.optim.Adam(
                self.model.get_generator_parameters(),
                lr=self.lr,
                weight_decay=0.0005
            )
            self.optimizer_classifier1 = torch.optim.Adam(
                self.model.get_classifier1_parameters(),
                lr=self.lr,
                weight_decay=0.0005
            )
            self.optimizer_classifier2 = torch.optim.Adam(
                self.model.get_classifier2_parameters(),
                lr=self.lr,
                weight_decay=0.0005
            )
            self.optimizer_decision_layer = torch.optim.Adam(
                self.model.get_decision_layer_parameters(),
                lr=self.lr,
                weight_decay=0.0005
            )

        if self.optimizer_type == 'SGD':
            self.lr = 0.001
            self.gamma = 10

            if self.task in ['AtoD', 'DtoW']:
                self.lr = 0.0003

            self.optimizer_generator = torch.optim.SGD(
                self.model.get_generator_parameters(),
                lr=self.lr,
                momentum=0.9,
                weight_decay=0.0005,
                nesterov=True
            )
            self.optimizer_classifier1 = torch.optim.SGD(
                self.model.get_classifier1_parameters(),
                lr=self.lr,
                momentum=0.9,
                weight_decay=0.0005,
                nesterov=True
            )
            self.optimizer_classifier2 = torch.optim.SGD(
                self.model.get_classifier2_parameters(),
                lr=self.lr,
                momentum=0.9,
                weight_decay=0.0005,
                nesterov=True
            )
            self.optimizer_decision_layer = torch.optim.SGD(
                self.model.get_decision_layer_parameters(),
                lr=self.lr,
                momentum=0.9,
                weight_decay=0.0005,
                nesterov=True
            )

    def update_optimizer(self, power=0.75, weight_decay=0.0005):
        """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
        if self.optimizer_type == 'SGD':
            if self.num_epochs != 999999:
                p = self.epoch / self.num_epochs
            else:
                p = self.iter_num / self.max_iter_num

            lr = self.lr * (1.0 + self.gamma * p) ** (-power)
        else:
            lr = self.lr

        self.cur_lr = lr

        for param_group in self.optimizer_generator.param_groups:
            param_group['lr'] = lr * param_group['lr_mult']
            param_group['weight_decay'] = weight_decay * param_group['decay_mult']

        for param_group in self.optimizer_classifier1.param_groups:
            param_group['lr'] = lr * param_group['lr_mult']
            param_group['weight_decay'] = weight_decay * param_group['decay_mult']

        for param_group in self.optimizer_classifier2.param_groups:
            param_group['lr'] = lr * param_group['lr_mult']
            param_group['weight_decay'] = weight_decay * param_group['decay_mult']

        for param_group in self.optimizer_decision_layer.param_groups:
            param_group['lr'] = lr * param_group['lr_mult']
            param_group['weight_decay'] = weight_decay * param_group['decay_mult']

    def reset_optimizer(self):
        self.optimizer_generator.zero_grad()
        self.optimizer_classifier1.zero_grad()
        self.optimizer_classifier2.zero_grad()

    def compute_discrepancy(self, output_t1, output_t2):
        return torch.mean(torch.abs(F.softmax(output_t1, dim=1) - F.softmax(output_t2, dim=1)))

    def train_one_epoch(self):
        since = time.time()
        self.model.train()

        source_corrects = 0

        total_source_num = len(self.data_loader['source']['train'].dataset)
        processed_source_num = 0

        for source_inputs, source_labels in self.data_loader['source']['train']:
            sys.stdout.write('\r{}/{}'.format(processed_source_num, total_source_num))
            sys.stdout.flush()

            self.update_optimizer()

            # TODO 1 : Step A

            self.reset_optimizer()

            source_inputs = source_inputs.to(self.device)

            source_outputs1, source_outputs2, decision_outputs = self.model(source_inputs)

            source_labels = source_labels.to(self.device)

            loss_source1 = nn.CrossEntropyLoss()(source_outputs1, source_labels)
            loss_source2 = nn.CrossEntropyLoss()(source_outputs2, source_labels)
            # loss_source3 = nn.CrossEntropyLoss()(decision_outputs, source_labels)
            # loss = loss_source1 + loss_source2 + loss_source3
            loss = loss_source1 + loss_source2

            loss.backward()
            self.optimizer_generator.step()
            self.optimizer_classifier1.step()
            self.optimizer_classifier2.step()
            # self.optimizer_decision_layer.step()
            self.reset_optimizer()

            _, source_class_preds = torch.max(decision_outputs, 1)
            source_corrects += (source_class_preds == source_labels.data).sum().item()
            processed_source_num += source_labels.size(0)

            # TODO 2 : Step B

            source_outputs1, source_outputs2, decision_outputs = self.model(source_inputs)

            loss_source1 = nn.CrossEntropyLoss()(source_outputs1, source_labels)
            loss_source2 = nn.CrossEntropyLoss()(source_outputs2, source_labels)
            # loss_source3 = nn.CrossEntropyLoss()(decision_outputs, source_labels)
            # loss_source = loss_source1 + loss_source2 + loss_source3
            # loss_source = loss_source1 + loss_source2 + loss_source3
            loss_source = loss_source1 + loss_source2

            target_iter = iter(self.data_loader['target']['train'])
            target_inputs, target_labels = next(target_iter)

            target_inputs = target_inputs.to(self.device)
            target_outputs1, target_outputs2, _ = self.model(target_inputs)
            loss_discrepancy = self.compute_discrepancy(target_outputs1, target_outputs2)

            loss = loss_source - loss_discrepancy
            loss.backward()
            self.optimizer_classifier1.step()
            self.optimizer_classifier2.step()
            # self.optimizer_decision_layer.step()
            self.reset_optimizer()

            # TODO 3 : Step C

            for i in range(self.num_k):
                target_outputs1, target_outputs2, _ = self.model(target_inputs)
                loss_discrepancy = self.compute_discrepancy(target_outputs1, target_outputs2)

                loss_discrepancy.backward()
                self.optimizer_generator.step()
                self.reset_optimizer()

            # TODO 5 : other parameters
            self.iter_num += 1

        acc = source_corrects / processed_source_num

        print()
        print('\nData size = {} , corrects = {}'.format(processed_source_num, source_corrects / 2))
        print('Using {:4f}'.format(time.time() - since))
        return 0, acc

from __future__ import print_function, division
import torch.nn as nn
import time
from data_helper import *
import sys
from solvers.Solver import Solver
from network import MT
import numpy as np
from torch.nn import functional as F
from torchvision import transforms
import random


class OldWeightEMA(object):
    """
    Exponential moving average weight optimizer for mean teacher model
    """

    def __init__(self, target_net, source_net, alpha=0.999):
        self.target_params = list(target_net.parameters())
        self.source_params = list(source_net.parameters())
        self.alpha = alpha

        for p, src_p in zip(self.target_params, self.source_params):
            p.data[:] = src_p.data[:]

    def step(self):
        one_minus_alpha = 1.0 - self.alpha
        for p, src_p in zip(self.target_params, self.source_params):
            p.data.mul_(self.alpha)
            p.data.add_(src_p.data * one_minus_alpha)


class MTSolver(Solver):

    def __init__(self, dataset_type, source_domain, target_domain, cuda='cuda:0',
                 pretrained=False,
                 batch_size=36,
                 num_epochs=9999, max_iter_num=9999999, test_interval=500, test_mode=False, num_workers=2,
                 clean_log=False, lr=0.001, gamma=10, loss_weight=3.0, optimizer_type='SGD', confidence_thresh=0.968,
                 rampup_epoch=80):
        super(MTSolver, self).__init__(
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
        self.model_name = 'MT'
        self.iter_num = 0
        self.loss_weight = loss_weight
        self.confidence_thresh = confidence_thresh
        self.rampup_epoch = rampup_epoch
        self.rampup_value = 0

    def set_model(self):
        if self.dataset_type == 'Digits':
            self.confidence_thresh = 0.968
            if self.task in ['MtoU', 'UtoM']:
                self.model = MT(n_classes=self.n_classes, base_model='DigitsMU')
            if self.task in ['StoM']:
                self.model = MT(n_classes=self.n_classes, base_model='DigitsStoM')

        if self.dataset_type == 'Office31':
            self.confidence_thresh = 0.90
            self.model = MT(n_classes=self.n_classes, base_model='ResNet50')

        if self.dataset_type == 'OfficeHome':
            self.confidence_thresh = 0.90
            self.model = MT(n_classes=self.n_classes, base_model='ResNet50')

        if self.pretrained:
            self.load_model(path=self.models_checkpoints_dir + '/' + self.model_name + '_best_train.pt')

        self.model = self.model.to(self.device)

    def test(self, data_loader):
        self.model.eval()

        total_loss = 0
        corrects = 0
        data_num = len(data_loader.dataset)
        processed_num = 0

        for inputs, labels in data_loader:
            sys.stdout.write('\r{}/{}'.format(processed_num, data_num))
            sys.stdout.flush()

            # inputs = self.augment(inputs, T=False, F=False)
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            # print('inputs ',inputs.size())
            # print('labels ',labels.size())

            class_outputs = self.model(source_x=inputs, test_mode=True)

            # print('class outputs ',class_outputs.size())

            _, preds = torch.max(class_outputs, 1)
            # print('preds ',preds.size())

            # loss = nn.CrossEntropyLoss()(class_outputs, labels)

            # total_loss += loss.item() * labels.size()[0]
            corrects += (preds == labels.data).sum().item()
            processed_num += labels.size()[0]

        acc = corrects / processed_num
        average_loss = total_loss / processed_num
        print('\nData size = {} , corrects = {}'.format(processed_num, corrects))

        return average_loss, acc

    def set_optimizer(self):
        super(MTSolver, self).set_optimizer()
        self.teacher_optimizer = OldWeightEMA(self.model.teacher, self.model.student)

    def compute_aug_loss(self, stu_out, tea_out):

        # conf_tea = torch.max(tea_out, 1)[0]
        # unsup_mask = (conf_tea > self.confidence_thresh).float()
        # unsup_mask_rate = unsup_mask.sum() / len(unsup_mask)

        d_aug_loss = stu_out - tea_out
        aug_loss = d_aug_loss * d_aug_loss
        aug_loss = aug_loss.mean(dim=1)

        # unsup_loss = (aug_loss * unsup_mask).mean()
        # return unsup_loss, unsup_mask_rate

        return aug_loss.mean() * self.rampup_value

    def augment(self, x, T=True, A=True):
        # tmp = torch.Tensor(x) + torch.randn_like(x) * 0.1
        # if random.random() < 0.5:
        #     tmp = tmp.flip([2])
        theta = np.zeros((x.size(0), 2, 3), dtype=np.float32)
        theta[:, 0, 0] = theta[:, 1, 1] = 1.0

        if T:
            theta[:, :, 2:] += np.random.uniform(low=-0.2, high=0.2, size=(x.size(0), 2, 1))

        if A:
            theta[:, :, :2] += np.random.normal(scale=0.1, size=(x.size(0), 2, 2))

        grid = F.affine_grid(theta=torch.from_numpy(theta), size=x.size())
        new_x = F.grid_sample(input=x, grid=grid)
        # print()
        # print(x.size())
        # print(x[0])
        # print()
        # print(new_x.size())
        # print(new_x[0])
        return new_x

    def train_one_epoch(self):

        since = time.time()
        self.model.train()

        total_loss = 0
        source_corrects = 0

        total_target_num = len(self.data_loader['target']['train'].dataset)
        processed_target_num = 0
        total_source_num = 0
        # CT_pass_rate = 0

        if self.epoch < self.rampup_epoch:
            p = max(0.0, float(self.epoch)) / float(self.rampup_epoch)
            p = 1.0 - p
            self.rampup_value = np.exp(-p * p * 5.0)
        else:
            self.rampup_value = 1.0
        print('ramup value = ', self.rampup_value)

        for target_inputs, target_labels in self.data_loader['target']['train']:
            sys.stdout.write('\r{}/{}'.format(processed_target_num, total_target_num))
            sys.stdout.flush()

            self.update_optimizer()

            self.optimizer.zero_grad()

            # TODO 1 : Target Train

            target_x1 = self.augment(target_inputs).to(self.device)
            target_x2 = self.augment(target_inputs).to(self.device)

            target_y1, target_y2 = self.model(target_x1=target_x1, target_x2=target_x2, test_mode=False,
                                              is_source=False)

            target_y1 = F.softmax(target_y1, dim=1)
            target_y2 = F.softmax(target_y2, dim=1)

            # aug_loss, CT_pass_rate = self.compute_aug_loss(target_y1, target_y2)
            aug_loss = self.compute_aug_loss(target_y1, target_y2)

            # TODO 1 : Source Train

            source_iter = iter(self.data_loader['source']['train'])
            source_inputs, source_labels = next(source_iter)
            source_inputs = self.augment(source_inputs).to(self.device)

            source_y = self.model(source_x=source_inputs, test_mode=False, is_source=True)
            source_labels = source_labels.to(self.device)

            class_loss = nn.CrossEntropyLoss()(source_y, source_labels)

            # TODO 3 : LOSS

            loss = class_loss + self.loss_weight * aug_loss

            loss.backward()

            self.optimizer.step()
            self.teacher_optimizer.step()

            # TODO 5 : other parameters
            total_loss += loss.item() * source_labels.size()[0]
            _, source_class_preds = torch.max(source_y, 1)
            source_corrects += (source_class_preds == source_labels.data).sum().item()
            total_source_num += source_labels.size()[0]
            processed_target_num += target_labels.size()[0]
            self.iter_num += 1

        acc = source_corrects / total_source_num
        average_loss = total_loss / total_source_num

        print()
        print('\nData size = {} , corrects = {}'.format(total_source_num, source_corrects))
        # print('CT pass rate : ', CT_pass_rate)
        print('Using {:4f}'.format(time.time() - since))

        return average_loss, acc

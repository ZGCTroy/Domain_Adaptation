from __future__ import print_function, division

import sys
import time

import torch.nn as nn

from data_helpers.data_helper import *
from networks.MYMADA import MYMADA
from solvers.Solver import Solver
import torch.nn.functional as F


class MYMADASolver(Solver):
    def __init__(self, dataset_type, source_domain, target_domain, cuda='cuda:0',
                 pretrained=False,
                 batch_size=32,
                 num_epochs=9999, max_iter_num=9999999, test_interval=500, test_mode=False, num_workers=2,
                 clean_log=False, lr=0.001, gamma=10, optimizer_type='SGD', loss_weight=1.0, data_root_dir='./data'):
        super(MYMADASolver, self).__init__(
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
            optimizer_type=optimizer_type,
            data_root_dir=data_root_dir
        )

        self.model_name = 'MYMADA_LossWeight10.0_BCEWeight_noEntropy_TargetClassLoss_-+'
        self.iter_num = 0
        self.class_weight = None
        self.loss_weight = loss_weight
        self.use_CT = False
        self.confidence_thresh = 0.97
        self.rampup_value = 1.0
        self.class_struct = None
        self.class_struct2 = None
        self.class_struct_first = None
        self.beta = 0.99
        self.targetClassLossWeight = 1.0

    def get_alpha(self, delta=10.0):
        if self.num_epochs != 999999:
            p = self.epoch / self.num_epochs
        else:
            p = self.iter_num / self.max_iter_num

        return np.float(2.0 / (1.0 + np.exp(-delta * p)) - 1.0)

    def set_model(self):
        if self.dataset_type == 'Digits':
            if self.task in ['MtoU', 'UtoM']:
                self.model = MYMADA(n_classes=self.n_classes, base_model='DigitsMU')
            if self.task in ['StoM']:
                self.model = MYMADA(n_classes=self.n_classes, base_model='DigitsStoM')

        if self.dataset_type in ['Office31', 'OfficeHome']:
            self.model = MYMADA(n_classes=self.n_classes, base_model='ResNet50')

        if self.pretrained:
            self.load_model(path=self.models_checkpoints_dir + '/' + self.model_name + '_best_train.pt')

        self.model = self.model.to(self.device)
        self.class_struct = [torch.zeros(size=(self.n_classes,), device=self.device) for i in range(self.n_classes)]
        self.class_struct2 = [torch.zeros(size=(self.n_classes,), device=self.device) for i in range(self.n_classes)]
        self.class_struct_first = [True for i in range(self.n_classes)]

    def test(self, data_loader, is_source=True, projection=False):
        model = self.model
        model.eval()

        total_loss = 0
        corrects = 0
        data_num = len(data_loader.dataset)
        processed_num = 0

        for inputs, labels in data_loader:
            sys.stdout.write('\r{}/{}'.format(processed_num, data_num))
            sys.stdout.flush()

            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            class_outputs = model(inputs, test_mode=True, is_source=is_source)

            _, preds = torch.max(class_outputs, 1)

            corrects += (preds == labels.data).sum().item()
            processed_num += labels.size()[0]

        acc = corrects / processed_num
        average_loss = total_loss / processed_num
        print('\nData size = {} , corrects = {}'.format(processed_num, corrects))
        print('loss weight =', self.loss_weight)

        return average_loss, acc

    def update_class_struct(self, source_class_outputs, alpha=0.9):
        # input: batch_size * n_classes
        alpha = self.beta
        source_soft_labels = torch.argmax(source_class_outputs.detach(), 1).detach()
        for i in range(source_class_outputs.size()[0]):
            label = source_soft_labels[i]
            # self.class_struct[label] = self.class_struct[label]*alpha + (1-alpha)*source_class_outputs[i]
            self.class_struct[label] += source_class_outputs[i]

        for i in range(self.n_classes):
            if torch.sum(self.class_struct[i]):
                self.class_struct2[i] = self.class_struct[i] / torch.sum(self.class_struct[i])

    def set_optimizer(self):
        if self.optimizer_type == 'Adam':
            self.optimizer_type = 'Adam'
            self.optimizer = torch.optim.Adam(
                params=self.model.get_parameters(),
                lr=self.lr,
                weight_decay=0.0005
            )
        else:
            self.optimizer_type = 'SGD'
            self.optimizer = torch.optim.SGD(
                self.model.get_parameters(),
                lr=self.lr,
                momentum=0.9,
                weight_decay=0.0005,
                nesterov=True
            )
            # self.generator_optimizer = torch.optim.SGD(
            #     self.model.get_generator_parameters(),
            #     lr=self.lr,
            #     momentum=0.9,
            #     weight_decay=0.0005,
            #     nesterov=True
            # )
            # self.classifier_optimizer = torch.optim.SGD(
            #     self.model.get_classifier_parameters(),
            #     lr=self.lr,
            #     momentum=0.9,
            #     weight_decay=0.0005,
            #     nesterov=True
            # )

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

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr * param_group['lr_mult']
            param_group['weight_decay'] = weight_decay * param_group['decay_mult']

        # for param_group in self.generator_optimizer.param_groups:
        #     param_group['lr'] = lr * param_group['lr_mult']
        #     param_group['weight_decay'] = weight_decay * param_group['decay_mult']
        #
        # for param_group in self.classifier_optimizer.param_groups:
        #     param_group['lr'] = lr * param_group['lr_mult']
        #     param_group['weight_decay'] = weight_decay * param_group['decay_mult']

    def train_one_epoch(self):
        since = time.time()
        self.model.train()

        total_loss = 0
        source_corrects = 0

        total_target_num = len(self.data_loader['target']['train'].dataset)
        processed_target_num = 0
        total_source_num = 0

        alpha = 0
        source_loss = 0
        target_loss = 0
        augment_loss = 0
        first = True

        for target_inputs, target_labels in self.data_loader['target']['train']:
            sys.stdout.write('\r{}/{}'.format(processed_target_num, total_target_num))
            sys.stdout.flush()

            self.update_optimizer()

            self.optimizer.zero_grad()

            alpha = self.get_alpha()
            self.rampup_value = alpha

            # TODO 1 : Source Train
            source_iter = iter(self.data_loader['source']['train'])
            source_inputs, source_labels = next(source_iter)
            batch_size = source_inputs.size()[0]
            source_inputs = source_inputs.to(self.device)

            source_domain_outputs, source_class_outputs = self.model(source_inputs, alpha=alpha, is_source=True)
            source_labels = source_labels.to(self.device)
            source_class_loss = nn.CrossEntropyLoss()(source_class_outputs, source_labels)
            source_class_outputs = nn.Softmax(dim=1)(source_class_outputs)

            # self.update_class_struct(source_class_outputs.detach())
            source_weight = self.get_weight(source_class_outputs.detach(), h=False)
            source_domain_loss = nn.BCELoss(weight=source_weight)(
                source_domain_outputs.view(-1),
                torch.zeros((batch_size * self.n_classes,), device=self.device)
            ) * self.loss_weight

            source_loss = source_class_loss + source_domain_loss
            source_loss.backward(retain_graph=True)
            self.optimizer.step()
            self.optimizer.zero_grad()

            # TODO 2 : Target Train
            target_inputs = target_inputs.to(self.device)

            batch_size = target_inputs.size()[0]

            target_domain_outputs, target_class_outputs = self.model(target_inputs, alpha=alpha, is_source=False)
            target_class_outputs = nn.Softmax(dim=1)(target_class_outputs)
            target_weight = self.get_weight(target_class_outputs.detach(), h=False)
            target_domain_loss = nn.BCELoss(weight=target_weight)(
                target_domain_outputs.view(-1),
                torch.ones((batch_size * self.n_classes,), device=self.device)
            ) * self.loss_weight

            target_domain_loss.backward(retain_graph=True)
            self.optimizer.step()
            self.optimizer.zero_grad()

            # TODO 3: Target Class Loss
            # target_soft_label = torch.argmax(target_class_outputs, 1)
            # target_max_value, _ = torch.max(target_class_outputs, 1)
            #
            # first = True
            # for label in target_soft_label:
            #     if first:
            #         class_struct = self.class_struct2[label].view(1, -1)
            #         first = False
            #     else:
            #         class_struct = torch.cat([class_struct.detach(), self.class_struct2[label].view(1, -1)], dim=0)
            # target_class_outputs = torch.log(target_class_outputs)
            # target_class_loss = torch.sum(torch.mul(target_class_outputs, class_struct.detach()), dim=1)
            # # target_class_loss = -torch.mean(target_class_loss)
            # target_class_loss = -torch.mean(
            #     torch.mul(target_class_loss, target_max_value.detach())) * self.targetClassLossWeight
            # # target_class_loss = 0
            #
            # target_class_loss.backward(retain_graph=True)
            # self.optimizer.step()
            # target_class_loss = -target_class_loss
            # target_class_loss.backward(retain_graph=True)
            # self.generator_optimizer.step()
            # self.generator_optimizer.zero_grad()
            # self.optimizer.zero_grad()
            # self.classifier_optimizer.zero_grad()
            target_class_loss = 0

            # TODO 5 : other parameters
            target_loss = target_class_loss + target_domain_loss
            loss = source_loss + target_loss
            total_loss += loss.item() * source_labels.size()[0]
            _, source_class_preds = torch.max(source_class_outputs, 1)
            source_corrects += (source_class_preds == source_labels.data).sum().item()
            total_source_num += source_labels.size()[0]
            processed_target_num += target_labels.size()[0]
            self.iter_num += 1

            self.writer.add_scalar('loss/class loss/source class loss', source_class_loss, self.iter_num)
            self.writer.add_scalar('loss/class loss/target class loss', target_class_loss, self.iter_num)
            self.writer.add_scalars('loss/class loss/group class loss', {
                'source class loss': source_class_loss,
                'target class loss': target_class_loss,
            }, self.iter_num
                                    )

            self.writer.add_scalar('loss/domain loss/source domain loss', source_domain_loss, self.iter_num)
            self.writer.add_scalar('loss/domain loss/target domain loss', target_domain_loss, self.iter_num)
            self.writer.add_scalars('loss/domain loss/group domain loss', {
                'source domain loss': source_domain_loss,
                'target domain loss': target_domain_loss,
            }, self.iter_num
                                    )

            # self.writer.add_scalar('loss/augment loss/augment class loss', augment_class_loss, self.iter_num)

            self.writer.add_scalar('parameters/alpha', alpha, self.iter_num)
            # self.writer.add_scalar('parameters/ramup value', self.rampup_value, self.iter_num)
            self.writer.add_scalar('parameters/Domain loss weight', self.loss_weight, self.iter_num)
            self.writer.add_scalar('parameters/target class loss weight', self.targetClassLossWeight, self.iter_num)

        acc = source_corrects / total_source_num
        average_loss = total_loss / total_source_num

        print()
        print('\nData size = {} , corrects = {}'.format(total_source_num, source_corrects))
        print('Using {:4f}'.format(time.time() - since))
        print('Alpha = ', alpha)
        print('source loss = {}, target_loss = {}, target augment loss = {}'.format(
            source_loss,
            target_loss,
            augment_loss
        ))

        return average_loss, acc

    def get_weight(self, x, h=True):
        if h:
            epsilon = 1e-5
            entropy = -x * torch.log(x + epsilon)
            entropy = torch.sum(entropy, dim=1)
            entropy = 1.0 + torch.exp(-entropy)
            entropy = entropy.view(-1, 1).repeat(1, self.n_classes).view(-1)
            weight = x.view(-1) * entropy
        else:
            weight = x.view(-1)

        return weight.detach()

    def compute_discrepancy(self, output_t1, output_t2):
        return torch.mean(torch.abs(output_t1 - output_t2))

    def augment(self, x, T=True, A=True):
        # tmp = torch.Tensor(x) + torch.randn_like(x) * 0.1

        N = x.size(0)
        theta = np.zeros((N, 2, 3), dtype=np.float32)
        theta[:, 0, 0] = theta[:, 1, 1] = 1.0

        if self.dataset_type in ['Office31', 'OfficeHome']:
            # hflip
            # theta_hflip = np.random.binomial(1, 0.5, size=(N,)) * 2 - 1
            # theta[:, 0, 0] = theta_hflip.astype(np.float32)
            #
            # # scale_u_range
            # scl = np.exp(
            #     np.random.uniform(
            #         low=np.log(0.75),
            #         high=np.log(1.33),
            #         size=(N,)
            #     )
            # )
            # theta[:, 0, 0] *= scl
            # theta[:, 1, 1] *= scl
            if T:
                theta[:, :, 2:] += np.random.uniform(low=-0.2, high=0.2, size=(N, 2, 1))

            if A:
                theta[:, :, :2] += np.random.normal(scale=0.1, size=(N, 2, 2))

        else:
            if T:
                theta[:, :, 2:] += np.random.uniform(low=-0.2, high=0.2, size=(N, 2, 1))

            if A:
                theta[:, :, :2] += np.random.normal(scale=0.1, size=(N, 2, 2))

        grid = F.affine_grid(theta=torch.from_numpy(theta), size=x.size())
        new_x = F.grid_sample(input=x, grid=grid)


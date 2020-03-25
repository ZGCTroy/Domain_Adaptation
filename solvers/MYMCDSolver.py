from __future__ import print_function, division

import sys
import time

import torch.nn as nn

from data_helpers.data_helper import *
from networks.MYMCD import MYMCD
from solvers.Solver import Solver
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import torchvision.utils as vutils

class MYMCDSolver(Solver):

    def __init__(self, dataset_type, source_domain, target_domain, cuda='cuda:0',
                 pretrained=False,
                 batch_size=36,
                 num_epochs=9999, max_iter_num=9999999, test_interval=500, test_mode=False, num_workers=2,
                 clean_log=False, lr=0.001, gamma=10, loss_weight=3.0, optimizer_type='SGD', num_k=4,data_root_dir='./data'):
        super(MYMCDSolver, self).__init__(
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
            data_root_dir=data_root_dir,
        )
        self.model_name = 'MYMCD_domain10_entropy_distance4_lowAlpha'
        self.iter_num = 0
        self.optimizer_generator = None
        self.optimizer_classifier1 = None
        self.optimizer_classifier2 = None
        self.num_k = num_k
        self.loss_weight = loss_weight
        self.lr = lr
        self.writer = SummaryWriter()

    def set_model(self):
        if self.dataset_type == 'Digits':
            if self.task in ['MtoU', 'UtoM']:
                self.model = MYMCD(n_classes=self.n_classes, base_model='DigitsMU')
            if self.task in ['StoM']:
                self.model = MYMCD(n_classes=self.n_classes, base_model='DigitsStoM')

        if self.dataset_type in ['Office31', 'OfficeHome']:
            self.model = MYMCD(n_classes=self.n_classes, base_model='ResNet50')

        if self.pretrained:
            self.load_model(path=self.models_checkpoints_dir + '/' + self.model_name + '_best_train.pt')

        self.model = self.model.to(self.device)

    def test(self, data_loader, projection=False):
        self.model.eval()

        corrects1 = 0
        corrects2 = 0
        corrects = 0
        data_num = len(data_loader.dataset)
        processed_num = 0
        first = True
        for inputs, labels in data_loader:

            sys.stdout.write('\r{}/{}'.format(processed_num, data_num))
            sys.stdout.flush()

            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            features, outputs2 = self.model(inputs, outputs1=False, outputs2=True, get_features=True)
            # outputs = nn.Softmax(dim=1)(outputs1) + nn.Softmax(dim=1)(outputs2)
            outputs = outputs2

            _, preds = torch.max(outputs, 1)
            corrects += (preds == labels.data).sum().item()

            processed_num += labels.size()[0]

            if first and projection:
                # print(features[:2].size())
                # x = vutils.make_grid(features[:2], normalize=True, scale_each=True)
                # print('save image')
                # print(features[0].size())
                # self.writer.add_image('feature map', features[0], self.epoch)

                self.writer.add_embedding(
                    features,
                    metadata=labels,
                    label_img=inputs,
                    global_step=self.epoch
                )


            first = False

        acc = corrects / processed_num
        print('\nData size = {} , corrects = {}'.format(processed_num, (corrects1 + corrects2) / 2))

        return 0, acc

    def set_optimizer(self):
        if self.optimizer_type == 'Adam':
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
            self.optimizer_domain_classifier = torch.optim.Adam(
                self.model.get_domain_classifier_parameters(),
                lr=self.lr,
                weight_decay=0.0005
            )

        if self.optimizer_type == 'SGD':
            self.gamma = 10

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
            self.optimizer_domain_classifier = torch.optim.SGD(
                self.model.get_domain_classifier_parameters(),
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

        for param_group in self.optimizer_domain_classifier.param_groups:
            param_group['lr'] = lr * param_group['lr_mult']
            param_group['weight_decay'] = weight_decay * param_group['decay_mult']

    def reset_optimizer(self):
        self.optimizer_generator.zero_grad()
        self.optimizer_classifier1.zero_grad()
        self.optimizer_classifier2.zero_grad()
        self.optimizer_domain_classifier.zero_grad()

    def compute_discrepancy(self, output_t1, output_t2):
        return torch.mean(torch.abs(F.softmax(output_t1, dim=1) - F.softmax(output_t2, dim=1)))

    def get_alpha(self, delta=10.0):
        if self.num_epochs != 999999:
            p = self.epoch / self.num_epochs
        else:
            p = self.iter_num / self.max_iter_num

        return np.float(2.0 / (1.0 + np.exp(-delta * p)) - 1.0)

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



        for target_inputs, target_labels in self.data_loader['target']['train']:
            sys.stdout.write('\r{}/{}'.format(processed_target_num, total_target_num))
            sys.stdout.flush()

            alpha = self.get_alpha()
            alpha = alpha * alpha * alpha
            self.update_optimizer()
            self.reset_optimizer()

            # TODO 1 : Source Train
            source_iter = iter(self.data_loader['source']['train'])
            source_inputs, source_labels = next(source_iter)
            batch_size = source_inputs.size()[0]

            source_domain_outputs, source_class_outputs = self.model(
                source_inputs.to(self.device),
                outputs1=True,
                outputs2=False,
                domain=True,
                alpha=alpha
            )
            source_class_loss = nn.CrossEntropyLoss()(source_class_outputs, source_labels.to(self.device))

            source_class_outputs = nn.Softmax(dim=1)(source_class_outputs)
            source_weight = self.get_weight(source_class_outputs.detach(), h=True).detach()
            source_domain_loss = nn.BCELoss(weight= source_weight)(
                source_domain_outputs.view(-1),
                torch.zeros((batch_size * self.n_classes,), device=self.device)
            )

            source_loss = source_class_loss + self.loss_weight * source_domain_loss

            source_loss.backward(retain_graph=True)
            self.optimizer_classifier1.step()
            self.optimizer_generator.step()
            self.optimizer_domain_classifier.step()
            self.reset_optimizer()

            # TODO 2 : Target Train
            batch_size = target_inputs.size()[0]

            target_domain_outputs, target_class_outputs1, target_class_outputs2 = self.model(
                target_inputs.to(self.device),
                outputs1=True,
                outputs2=True,
                domain=True,
                alpha=alpha
            )

            target_soft_labels = torch.argmax(target_class_outputs1.detach(), 1).detach()
            target_class_loss = nn.CrossEntropyLoss()(target_class_outputs2, target_soft_labels)

            target_class_outputs = nn.Softmax(dim=1)(target_class_outputs2)
            target_weight = self.get_weight(target_class_outputs.detach(), h=True).detach()
            target_domain_loss = nn.BCELoss(weight=target_weight)(
                target_domain_outputs.view(-1),
                torch.ones((batch_size * self.n_classes,), device=self.device)
            )

            target_loss = target_class_loss + self.loss_weight * target_domain_loss

            target_loss.backward(retain_graph=True)
            self.optimizer_classifier2.step()
            self.optimizer_generator.step()
            self.optimizer_domain_classifier.step()
            self.reset_optimizer()

            target_class_outputs1, target_class_outputs2 = self.model(
                target_inputs,
                outputs1=True,
                outputs2=True
            )
            target_soft_labels = torch.argmax(target_class_outputs1.detach(), 1).detach()
            target_class_loss = nn.CrossEntropyLoss()(target_class_outputs2, target_soft_labels)
            loss_discrepancy = self.compute_discrepancy(
                target_class_outputs1,
                target_class_outputs2
            )
            loss = target_class_loss - loss_discrepancy
            loss.backward(retain_graph=True)
            self.optimizer_classifier1.step()
            self.optimizer_classifier2.step()
            self.reset_optimizer()

            # TODO 3 : Distance Loss
            for i in range(self.num_k):
                target_class_outputs1, target_class_outputs2 = self.model(
                    target_inputs,
                    outputs1=True,
                    outputs2=True
                )
                loss_discrepancy = self.compute_discrepancy(
                    target_class_outputs1,
                    target_class_outputs2
                )

                loss = loss_discrepancy
                loss.backward(retain_graph=True)
                self.optimizer_generator.step()
                self.reset_optimizer()

            loss = source_loss + target_loss
            # TODO 5 : other parameters
            total_loss += loss.item() * source_labels.size()[0]
            _, source_class_preds = torch.max(source_class_outputs, 1)
            source_corrects += (source_class_preds == source_labels.data).sum().item()
            total_source_num += source_labels.size()[0]
            processed_target_num += target_labels.size()[0]
            self.iter_num += 1

            self.writer.add_scalar('loss/class loss/source class loss',source_class_loss, self.iter_num)
            self.writer.add_scalar('loss/class loss/target class loss', target_class_loss, self.iter_num)
            self.writer.add_scalars('loss/class loss/group class loss',{
                    'source class loss':source_class_loss,
                    'target class loss':target_class_loss,
                }, self.iter_num
            )
            self.writer.add_scalar('loss/domain loss/source domain loss', source_domain_loss, self.iter_num)
            self.writer.add_scalar('loss/domain loss/target domain loss', target_domain_loss, self.iter_num)
            self.writer.add_scalars('loss/domain loss/group domain loss', {
                'source domain loss': source_domain_loss,
                'target domain loss': target_domain_loss,
            }, self.iter_num
                                    )

            self.writer.add_scalar('loss/distance loss', loss_discrepancy, self.iter_num)
            self.writer.add_scalar('alpha', alpha, self.iter_num)





        acc = source_corrects / total_source_num
        average_loss = total_loss / total_source_num

        print()
        print('\nData size = {} , corrects = {}'.format(total_source_num, source_corrects))
        print('Using {:4f}'.format(time.time() - since))
        print('Alpha = ', alpha)
        print("Loss weight = ", self.loss_weight)
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
        return weight * self.n_classes



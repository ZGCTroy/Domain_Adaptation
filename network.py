import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Function
from data_helper import *
from torch.utils import data
import torchvision


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)


class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None


# BaseLine network for "SVHN [3,32,32] -> MNIST [3,32,32]"
class DigitsStoM(nn.Module):
    def __init__(self, n_classes):
        super(DigitsStoM, self).__init__()

        self.feature_extracter = nn.Sequential(
            nn.Conv2d(3, 128, (3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, (3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, (3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Dropout(),
            nn.Conv2d(128, 256, (3, 3), padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, (3, 3), padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, (3, 3), padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Dropout(),
            nn.Conv2d(256, 512, (3, 3), padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 256, (1, 1), padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 128, (1, 1), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AvgPool2d((6, 6))
        )

        self.features_output_size = 128

        self.classifier = nn.Sequential(
            nn.Linear(128, n_classes)
        )

    def forward(self, x, get_features=False, get_class_outputs=True):
        if get_features == False and get_class_outputs == False:
            return None

        features = self.feature_extracter(x)
        features = features.view(-1, 128)

        if get_features == True and get_class_outputs == False:
            return features

        class_outputs = self.classifier(features)

        if get_features:
            return features, class_outputs
        else:
            return class_outputs

    def get_parameters(self):
        parameters = [
            {'params': self.feature_extracter.parameters(), 'lr_mult': 1, 'decay_mult': 1},
            {'params': self.classifier.parameters(), 'lr_mult': 1, 'decay_mult': 1}
        ]
        return parameters


# BaseLine network for "MNIST [1,28,28] <-> USPS[1,28,28]"
class DigitsMU(nn.Module):
    def __init__(self, n_classes):
        super(DigitsMU, self).__init__()
        self.n_classes = n_classes

        self.feature_extracter = nn.Sequential(
            nn.Conv2d(1, 32, (5, 5)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(32, 64, (3, 3)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3, 3)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Dropout()
        )

        self.features_output_size = 1024

        self.classifier = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, n_classes)
        )

    def forward(self, x, get_features=False, get_class_outputs=True):
        if get_features == False and get_class_outputs == False:
            return None

        features = self.feature_extracter(x)
        features = features.view(-1, 1024)

        if get_features == True and get_class_outputs == False:
            return features

        class_outputs = self.classifier(features)

        if get_features:
            return features, class_outputs
        else:
            return class_outputs

    def get_parameters(self):
        return [
            {'params': self.feature_extracter.parameters(), 'lr_mult': 1, 'decay_mult': 1},
            {'params': self.classifier.parameters(), 'lr_mult': 1, 'decay_mult': 1}
        ]


class AdversarialNetwork(nn.Module):
    def __init__(self, in_feature, hidden_size, lr_mult=10, decay_mult=2, output_num=1, sigmoid=True):
        super(AdversarialNetwork, self).__init__()
        self.in_feature = in_feature
        self.hidden_size = hidden_size
        self.lr_mult = lr_mult
        self.decay_mult = decay_mult

        self.discriminator = nn.Sequential(
            nn.Linear(in_feature, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_size, output_num),
            # nn.Sigmoid()
        )
        if sigmoid:
            self.discriminator.add_module(name='sigmoid', module=nn.Sigmoid())

        self.output_num = output_num

    def forward(self, x, alpha):
        x = ReverseLayerF.apply(x, alpha)

        y = self.discriminator(x)

        return y

    def get_parameters(self):
        parameters = [
            {"params": self.discriminator.parameters(), "lr_mult": self.lr_mult, 'decay_mult': self.decay_mult}
        ]
        return parameters


class ResNet50(nn.Module):
    def __init__(self, bottleneck_dim=256, n_classes=1000, pretrained=True):
        super(ResNet50, self).__init__()
        self.n_classes = n_classes
        self.pretrained = pretrained

        resnet50 = torchvision.models.resnet50(pretrained=pretrained)

        # Extracter
        self.feature_extracter = nn.Sequential(
            resnet50.conv1,
            resnet50.bn1,
            resnet50.relu,
            resnet50.maxpool,
            resnet50.layer1,
            resnet50.layer2,
            resnet50.layer3,
            resnet50.layer4,
            resnet50.avgpool,
        )

        self.bottleneck = nn.Linear(resnet50.fc.in_features, bottleneck_dim)
        self.bottleneck.apply(init_weights)
        self.features_output_size = bottleneck_dim

        # Class Classifier
        self.fc = nn.Linear(bottleneck_dim, n_classes)
        self.fc.apply(init_weights)
        self.__in_features = bottleneck_dim
        self.classifier = nn.Sequential(
            self.fc
        )

    def forward(self, x, get_features=False, get_class_outputs=True):
        if get_features == False and get_class_outputs == False:
            return None
        features = self.feature_extracter(x)
        features = features.view(features.size(0), -1)
        features = self.bottleneck(features)

        if get_features == True and get_class_outputs == False:
            return features
        class_outputs = self.classifier(features)

        if get_features:
            return features, class_outputs
        else:
            return class_outputs

    def get_parameters(self):
        parameters = [
            {'params': self.feature_extracter.parameters(), 'lr_mult': 1, 'decay_mult': 1},
            {'params': self.bottleneck.parameters(), 'lr_mult': 10, 'decay_mult': 2},
            {'params': self.classifier.parameters(), 'lr_mult': 10, 'decay_mult': 2}
        ]

        return parameters


class DANN(nn.Module):
    def __init__(self, n_classes, base_model, pretrained=True):
        super(DANN, self).__init__()

        self.n_classes = n_classes
        self.pretrained = pretrained

        if base_model == 'ResNet50':
            self.base_model = ResNet50(n_classes=n_classes, pretrained=pretrained)
            self.lr_mult = 10
            self.decay_mult = 2

        if base_model == 'DigitsStoM':
            self.base_model = DigitsStoM(n_classes=n_classes)
            self.lr_mult = 1
            self.decay_mult = 1

        if base_model == 'DigitsMU':
            self.base_model = DigitsMU(n_classes=n_classes)
            self.lr_mult = 1
            self.decay_mult = 1

        self.domain_classifier = AdversarialNetwork(
            in_feature=self.base_model.features_output_size,
            hidden_size=1024,
            lr_mult=self.lr_mult,
            decay_mult=self.decay_mult
        )

    def forward(self, x, alpha=1.0, test_mode=False, is_source=True):
        if test_mode:
            class_outputs = self.base_model(x, get_features=False, get_class_outputs=True)
            return class_outputs

        if is_source:
            features, class_outputs = self.base_model(x, get_features=True, get_class_outputs=True)
            domain_outputs = self.domain_classifier(features, alpha=alpha)
            return domain_outputs, class_outputs
        else:
            features = self.base_model(x, get_features=True, get_class_outputs=False)
            domain_outputs = self.domain_classifier(features, alpha=alpha)
            return domain_outputs

    def get_parameters(self):
        return self.base_model.get_parameters() + self.domain_classifier.get_parameters()


class MT(nn.Module):
    def __init__(self, n_classes, base_model, pretrained=True):
        super(MT, self).__init__()

        self.n_classes = n_classes
        self.pretrained = pretrained

        if base_model == 'ResNet50':
            self.student = ResNet50(n_classes=n_classes, pretrained=pretrained)
            self.teacher = ResNet50(n_classes=n_classes, pretrained=pretrained)

        if base_model == 'DigitsStoM':
            self.student = DigitsStoM(n_classes=n_classes)
            self.teacher = DigitsStoM(n_classes=n_classes)

        if base_model == 'DigitsMU':
            self.student = DigitsMU(n_classes=n_classes)
            self.teacher = DigitsMU(n_classes=n_classes)

        for param in self.teacher.parameters():
            param.requires_grad = False

    def forward(self, source_x=None, target_x1=None, target_x2=None, test_mode=False, is_source=True):
        if test_mode:
            class_outputs = self.student(source_x, get_features=False, get_class_outputs=True)
            return class_outputs

        if is_source:
            source_y = self.student(source_x, get_features=False, get_class_outputs=True)
            return source_y
        else:
            target_y1 = self.student(target_x1, get_features=False, get_class_outputs=True)
            target_y2 = self.teacher(target_x1, get_features=False, get_class_outputs=True)
            return target_y1, target_y2

    def get_parameters(self):
        return self.student.get_parameters()


class MCD(nn.Module):
    def __init__(self, n_classes, base_model, pretrained=True):
        super(MCD, self).__init__()

        self.n_classes = n_classes
        self.pretrained = pretrained
        self.base_model_name = base_model

        if base_model == 'ResNet50':
            self.Generator = ResNet50(n_classes=n_classes, pretrained=pretrained)
            self.Classifier1 = nn.Sequential(
                nn.Linear(256, n_classes)
            )
            self.Classifier2 = nn.Sequential(
                nn.Linear(256, n_classes)
            )
            self.lr_mult = 10
            self.decay_mult = 2

        if base_model == 'DigitsStoM':
            self.Generator = DigitsStoM(n_classes=n_classes)
            self.Classifier1 = nn.Sequential(
                nn.Linear(128, n_classes)
            )
            self.Classifier2 = nn.Sequential(
                nn.Linear(128, n_classes)
            )
            self.lr_mult = 1
            self.decay_mult = 1

        if base_model == 'DigitsMU':
            self.Generator = DigitsMU(n_classes=n_classes)
            self.Classifier1 = nn.Sequential(
                nn.Linear(1024, 256),
                nn.Linear(256, n_classes)
            )
            self.Classifier2 = nn.Sequential(
                nn.Linear(1024, 256),
                nn.Linear(256, n_classes)
            )
            self.lr_mult = 1
            self.decay_mult = 1

    def forward(self, x):
        features = self.Generator(x, get_features=True, get_class_outputs=False)
        outputs1 = self.Classifier1(features)
        outputs2 = self.Classifier2(features)

        return outputs1, outputs2

    def get_generator_parameters(self):
        if self.base_model_name == 'ResNet50':
            parameters = [
                {'params': self.Generator.feature_extracter.parameters(), 'lr_mult': 1, 'decay_mult': 1},
                {'params': self.Generator.bottleneck.parameters(), 'lr_mult': 10, 'decay_mult': 2},
            ]
            return parameters
        else:
            parameters = [
                {'params': self.Generator.feature_extracter.parameters(), 'lr_mult': 1, 'decay_mult': 1},
            ]
            return parameters

    def get_classifier1_parameters(self):
        parameters = [
            {'params': self.Classifier1.parameters(), 'lr_mult': self.lr_mult, 'decay_mult': self.decay_mult},
        ]
        return parameters

    def get_classifier2_parameters(self):
        parameters = [
            {'params': self.Classifier2.parameters(), 'lr_mult': self.lr_mult, 'decay_mult': self.decay_mult},
        ]
        return parameters


class SmallAdversarialNetwork(nn.Module):
    def __init__(self, in_feature, hidden_size, lr_mult=10, decay_mult=2, output_num=1, sigmoid=True):
        super(SmallAdversarialNetwork, self).__init__()
        self.in_feature = in_feature
        self.hidden_size = hidden_size
        self.lr_mult = lr_mult
        self.decay_mult = decay_mult

        self.discriminator = nn.Sequential(
            nn.Linear(in_feature, 1024),
            nn.ReLU(),
            nn.Linear(1024,output_num)
            # nn.Sigmoid()
        )
        if sigmoid:
            self.discriminator.add_module(name='sigmoid', module=nn.Sigmoid())

        self.output_num = output_num

    def forward(self, x, alpha):
        x = ReverseLayerF.apply(x, alpha)

        y = self.discriminator(x)

        return y

    def get_parameters(self):
        parameters = [
            {"params": self.discriminator.parameters(), "lr_mult": self.lr_mult, 'decay_mult': self.decay_mult}
        ]
        return parameters


class MADA(nn.Module):
    def __init__(self, n_classes, base_model, pretrained=True):
        super(MADA, self).__init__()

        self.n_classes = n_classes
        self.pretrained = pretrained

        if base_model == 'ResNet50':
            self.base_model = ResNet50(n_classes=n_classes, pretrained=pretrained)
            self.lr_mult = 10
            self.decay_mult = 2

        if base_model == 'DigitsStoM':
            self.base_model = DigitsStoM(n_classes=n_classes)
            self.lr_mult = 1
            self.decay_mult = 1

        if base_model == 'DigitsMU':
            self.base_model = DigitsMU(n_classes=n_classes)
            self.lr_mult = 1
            self.decay_mult = 1

        self.domain_classifiers = nn.ModuleList()
        for i in range(n_classes):
            self.domain_classifiers.append(
                AdversarialNetwork(
                    in_feature=self.base_model.features_output_size,
                    hidden_size=1024,
                    lr_mult=self.lr_mult,
                    decay_mult=self.decay_mult,
                    output_num=1,
                    sigmoid=True
                )
            )

    def forward(self, x, alpha=1.0, test_mode=False):
        if test_mode:
            class_outputs = self.base_model(x, get_features=False, get_class_outputs=True)
            return class_outputs

        features, class_outputs = self.base_model(x, get_features=True, get_class_outputs=True)

        softmax_class_outputs = nn.Softmax(dim=1)(class_outputs).detach()

        i = -1
        domain_outputs = []
        for ad in self.domain_classifiers:
            i += 1
            weighted_features = softmax_class_outputs[:, i].view(-1, 1) * features
            if i == 0:
                domain_outputs = ad(weighted_features, alpha=alpha)
            else:
                domain_outputs = torch.cat([domain_outputs, ad(weighted_features, alpha=alpha)], dim=1)

        return domain_outputs, class_outputs

    def get_parameters(self):
        parameters = self.base_model.get_parameters()
        for ad in self.domain_classifiers:
            parameters += ad.get_parameters()

        return parameters


def main():
    Amazon = load_Office(root_dir='./data/Office31', domain='Amazon')
    data_loader = data.DataLoader(
        dataset=Amazon['train'],
        batch_size=16,
        shuffle=False,
        num_workers=0
    )

    images, labels = iter(data_loader).next()
    print(images.size(), labels.size())

    dann = DANN(n_classes=31)

    domain_outputs, class_outputs = dann(images)

    print('domain outputs ', domain_outputs.size())
    print('class outputs ', class_outputs.size())


if __name__ == '__main__':
    main()

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

        self.conv1_1 = nn.Conv2d(3, 128, (3, 3), padding=1)
        self.conv1_1_bn = nn.BatchNorm2d(128)
        self.conv1_2 = nn.Conv2d(128, 128, (3, 3), padding=1)
        self.conv1_2_bn = nn.BatchNorm2d(128)
        self.conv1_3 = nn.Conv2d(128, 128, (3, 3), padding=1)
        self.conv1_3_bn = nn.BatchNorm2d(128)
        self.pool1 = nn.MaxPool2d((2, 2))
        self.drop1 = nn.Dropout()

        self.conv2_1 = nn.Conv2d(128, 256, (3, 3), padding=1)
        self.conv2_1_bn = nn.BatchNorm2d(256)
        self.conv2_2 = nn.Conv2d(256, 256, (3, 3), padding=1)
        self.conv2_2_bn = nn.BatchNorm2d(256)
        self.conv2_3 = nn.Conv2d(256, 256, (3, 3), padding=1)
        self.conv2_3_bn = nn.BatchNorm2d(256)
        self.pool2 = nn.MaxPool2d((2, 2))
        self.drop2 = nn.Dropout()

        self.conv3_1 = nn.Conv2d(256, 512, (3, 3), padding=0)
        self.conv3_1_bn = nn.BatchNorm2d(512)
        self.nin3_2 = nn.Conv2d(512, 256, (1, 1), padding=1)
        self.nin3_2_bn = nn.BatchNorm2d(256)
        self.nin3_3 = nn.Conv2d(256, 128, (1, 1), padding=1)
        self.nin3_3_bn = nn.BatchNorm2d(128)

        self.features_output_size = 128

        self.fc4 = nn.Linear(128, n_classes)

    def forward(self, x, get_features=False, get_class_outputs=True):
        if get_features == False and get_class_outputs == False:
            return None

        x = F.relu(self.conv1_1_bn(self.conv1_1(x)))
        x = F.relu(self.conv1_2_bn(self.conv1_2(x)))
        x = self.pool1(F.relu(self.conv1_3_bn(self.conv1_3(x))))
        x = self.drop1(x)

        x = F.relu(self.conv2_1_bn(self.conv2_1(x)))
        x = F.relu(self.conv2_2_bn(self.conv2_2(x)))
        x = self.pool2(F.relu(self.conv2_3_bn(self.conv2_3(x))))
        x = self.drop2(x)

        x = F.relu(self.conv3_1_bn(self.conv3_1(x)))
        x = F.relu(self.nin3_2_bn(self.nin3_2(x)))
        x = F.relu(self.nin3_3_bn(self.nin3_3(x)))

        x = F.avg_pool2d(x, 6)
        features = x.view(-1, 128)

        if get_features == True and get_class_outputs == False:
            return features

        class_outputs = self.fc4(features)

        if get_features == True:
            return features, class_outputs
        else:
            return class_outputs

    def get_parameters(self):
        return [{'params': self.parameters(), 'lr_mult': 1, 'decay_mult': 1}]


# BaseLine network for "MNIST [1,28,28] <-> USPS[1,28,28]"
class DigitsMU(nn.Module):
    def __init__(self, n_classes):
        super(DigitsMU, self).__init__()
        self.n_classes = n_classes
        self.conv1_1 = nn.Conv2d(1, 32, (5, 5))
        self.conv1_1_bn = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d((2, 2))

        self.conv2_1 = nn.Conv2d(32, 64, (3, 3))
        self.conv2_1_bn = nn.BatchNorm2d(64)
        self.conv2_2 = nn.Conv2d(64, 64, (3, 3))
        self.conv2_2_bn = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d((2, 2))

        self.features_output_size = 1024

        self.drop1 = nn.Dropout()
        self.fc3 = nn.Linear(1024, 256)
        self.fc4 = nn.Linear(256, n_classes)

    def forward(self, x, get_features=False, get_class_outputs=True):
        if get_features == False and get_class_outputs == False:
            return None

        x = self.pool1(F.relu(self.conv1_1_bn(self.conv1_1(x))))
        x = F.relu(self.conv2_1_bn(self.conv2_1(x)))
        x = self.pool2(F.relu(self.conv2_2_bn(self.conv2_2(x))))
        x = x.view(-1, 1024)
        features = self.drop1(x)

        if get_features == True and get_class_outputs == False:
            return features

        class_outputs = self.fc4(F.relu(self.fc3(features)))

        if get_features == True:
            return features, class_outputs
        else:
            return class_outputs

    def get_parameters(self):
        return [{'params': self.parameters(), 'lr_mult': 1, 'decay_mult': 1}]


class AdversarialNetwork(nn.Module):
    def __init__(self, in_feature, hidden_size, lr_mult=10, decay_mult=2):
        super(AdversarialNetwork, self).__init__()
        self.in_feature = in_feature
        self.hidden_size = hidden_size
        self.lr_mult = lr_mult
        self.decay_mult = decay_mult

        self.classifier = nn.Sequential(
            nn.Linear(in_feature, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_size, 1),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.classifier(x)
        # y = y.view(-1)
        return y

    def output_num(self):
        return 1

    def get_parameters(self):
        return [{"params": self.parameters(), "lr_mult": self.lr_mult, 'decay_mult': self.decay_mult}]


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

    def forward(self, x, get_features=False, get_class_outputs=True):
        if get_features == False and get_class_outputs == False:
            return None
        features = self.feature_extracter(x)
        features = features.view(features.size(0), -1)
        features = self.bottleneck(features)

        if get_features == True and get_class_outputs == False:
            return features
        class_outputs = self.fc(features)

        if get_features == True:
            return features, class_outputs
        else:
            return class_outputs

    def get_parameters(self):
        parameters = [
            {'params': self.feature_extracter.parameters(), 'lr_mult': 1, 'decay_mult': 2},
            {'params': self.bottleneck.parameters(), 'lr_mult': 10, 'decay_mult': 2},
            {'params': self.fc.parameters(), 'lr_mult': 10, 'decay_mult': 2}
        ]

        return parameters


class DANN(nn.Module):
    def __init__(self, n_classes, base_model, pretrained=True):
        super(DANN, self).__init__()

        self.n_classes = n_classes
        self.pretrained = pretrained

        if base_model == 'ResNet50':
            self.base_model = ResNet50(n_classes=n_classes, pretrained=pretrained)
            self.domain_classifier = AdversarialNetwork(
                in_feature=self.base_model.features_output_size,
                hidden_size=1024,
                lr_mult=10,
                decay_mult=2
            )

        if base_model == 'DigitsStoM':
            self.base_model = DigitsStoM(n_classes=n_classes)
            self.domain_classifier = AdversarialNetwork(
                in_feature=self.base_model.features_output_size,
                hidden_size=1024,
                lr_mult=1,
                decay_mult=1
            )

        if base_model == 'DigitsMU':
            self.base_model = DigitsMU(n_classes=n_classes)
            self.domain_classifier = AdversarialNetwork(
                in_feature=self.base_model.features_output_size,
                hidden_size=1024,
                lr_mult=1,
                decay_mult=1
            )

    def forward(self, x, alpha=1.0, test_mode=False, is_source=True):
        if test_mode == True:
            class_outputs = self.base_model(x, get_features=False, get_class_outputs=True)
            return class_outputs

        if is_source:
            features, class_outputs = self.base_model(x, get_features=True, get_class_outputs=True)
            features = ReverseLayerF.apply(features, alpha)
            domain_outputs = self.domain_classifier(features)
            return domain_outputs, class_outputs
        else:
            features = self.base_model(x, get_features=True, get_class_outputs=False)
            features = ReverseLayerF.apply(features, alpha)
            domain_outputs = self.domain_classifier(features)
            return domain_outputs

    def get_parameters(self):
        return self.base_model.get_parameters() + self.domain_classifier.get_parameters()


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

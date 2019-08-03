import torchvision
from torch import nn


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


def get_small_classifier(in_features_size, n_classes):
    small_classifier = nn.Sequential(
        nn.Linear(in_features_size, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Linear(256, n_classes),
    )
    return small_classifier


def get_large_classifier(in_features_size, n_classes):
    classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_features_size, 1024),
        nn.BatchNorm1d(1024),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(1024, 1024),
        nn.BatchNorm1d(1024),
        nn.ReLU(),
        nn.Linear(1024, n_classes)
    )

    return classifier


# Domain_Adaptation network for "MNIST [1,28,28] <-> USPS[1,28,28]"
class DigitsMU(nn.Module):
    def __init__(self, n_classes, use_dropout=False):
        super(DigitsMU, self).__init__()
        self.n_classes = n_classes
        self.use_dropout = use_dropout

        self.normalization_layer = nn.BatchNorm2d(1)

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
        )

        self.use_dropout = True
        if self.use_dropout:
            self.feature_extracter.add_module(name='dropout', module=nn.Dropout(0.5))

        self.features_output_size = 1024

        self.classifier = get_small_classifier(
            in_features_size=self.features_output_size,
            n_classes=n_classes
        )

    def forward(self, x, get_features=False, get_class_outputs=True):
        if get_features == False and get_class_outputs == False:
            return None

        x = self.normalization_layer(x)

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


# Domain_Adaptation network for "SVHN [3,32,32] -> MNIST [3,32,32]"
class DigitsStoM(nn.Module):
    def __init__(self, n_classes, use_dropout=False):
        super(DigitsStoM, self).__init__()

        self.use_dropout = use_dropout
        self.normalization_layer = nn.BatchNorm2d(3)

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

        if self.use_dropout:
            self.feature_extracter.add_module(name='dropout', module=nn.Dropout(0.5))

        self.features_output_size = 128

        self.classifier = get_large_classifier(
            in_features_size=self.features_output_size,
            n_classes=n_classes
        )

    def forward(self, x, get_features=False, get_class_outputs=True):
        if get_features == False and get_class_outputs == False:
            return None

        x = self.normalization_layer(x)

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


# ResNet for Office31 and OfficeHome
class ResNet50(nn.Module):
    def __init__(self, bottleneck_dim=256, n_classes=1000, pretrained=True, use_dropout=False):
        super(ResNet50, self).__init__()
        self.n_classes = n_classes
        self.pretrained = pretrained
        self.use_dropout = use_dropout

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

        if use_dropout:
            self.dropout = nn.Dropout(0.5)

        # Class Classifie
        self.classifier = nn.Sequential(
            nn.Linear(self.features_output_size, n_classes)
        )
        self.classifier.apply(init_weights)

    def forward(self, x, get_features=False, get_class_outputs=True):
        if get_features == False and get_class_outputs == False:
            return None
        features = self.feature_extracter(x)
        features = features.view(features.size(0), -1)
        features = self.bottleneck(features)

        if self.use_dropout:
            features = self.dropout(features)

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

from networks.AdversarialNetwork import AdversarialNetwork
from networks.Baseline import *


class RTNMADA(nn.Module):
    def __init__(self, n_classes, base_model, pretrained=True):
        super(RTNMADA, self).__init__()

        self.n_classes = n_classes
        self.pretrained = pretrained

        if base_model == 'ResNet50':
            self.base_model = ResNet50(n_classes=n_classes, pretrained=pretrained, bottleneck_dim=256, use_dropout=False)
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
            in_features_size=self.base_model.features_output_size,
            lr_mult=self.lr_mult,
            decay_mult=self.decay_mult,
            out_features_size=n_classes,
            sigmoid=True
        )

        self.RTN = nn.Sequential(
            nn.Linear(self.n_classes, 128),
            nn.Linear(128, self.n_classes)
        )


    def forward(self, x, alpha=1.0, is_source=True, test_mode=False):
        if test_mode:
            class_outputs = self.base_model(x, get_features=False, get_class_outputs=True)
            if is_source:
                class_outputs = class_outputs + self.RTN(class_outputs)
            return class_outputs

        features, class_outputs = self.base_model(x, get_features=True, get_class_outputs=True)

        if is_source:
            class_outputs = class_outputs + self.RTN(class_outputs)

        domain_outputs = self.domain_classifier(features, alpha=alpha)

        return domain_outputs, class_outputs

    def get_parameters(self):
        parameters = self.get_generator_parameters() + self.get_classifier_parameters() + self.get_discriminator_parameters()
        return parameters

    def get_generator_parameters(self):
        parameters = self.base_model.get_generator_parameters()
        return parameters

    def get_classifier_parameters(self):
        parameters = self.base_model.get_classifier_parameters() + self.get_RTN_parameters()
        return parameters

    def get_discriminator_parameters(self):
        parameters = self.domain_classifier.get_parameters()
        return parameters

    def get_RTN_parameters(self):
        parameters = [
            {'params': self.RTN.parameters(), 'lr_mult': 10, 'decay_mult': 2},
        ]
        return parameters


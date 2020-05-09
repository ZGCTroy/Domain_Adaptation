from networks.AdversarialNetwork import AdversarialNetwork
from networks.Baseline import *


class CycleMADA(nn.Module):
    def __init__(self, n_classes, base_model, pretrained=True):
        super(CycleMADA, self).__init__()

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
        self.RTN1 = nn.Linear(self.n_classes, 128)
        self.RTN2 = nn.Linear(self.n_classes, 128)
        self.RTN3 = nn.Linear(128, self.n_classes)

    def forward(self, x, alpha=1.0, is_source=True, test_mode=False):
        features, class_outputs = self.base_model(x, get_features=True, get_class_outputs=True)

        domain_outputs = self.domain_classifier(features, alpha=alpha)

        class_outputs = self.RTN3(
            self.RTN1(class_outputs)+self.RTN2(domain_outputs.detach())
        )
        # class_outputs = self.RTN(class_outputs * domain_outputs.detach())

        if test_mode:
            return class_outputs

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
            {'params': self.RTN.parameters(), 'lr_mult': self.lr_mult, 'decay_mult': self.decay_mult},
            {'params': self.RTN1.parameters(), 'lr_mult': self.lr_mult, 'decay_mult': self.decay_mult},
            {'params': self.RTN2.parameters(), 'lr_mult': self.lr_mult, 'decay_mult': self.decay_mult},
            {'params': self.RTN3.parameters(), 'lr_mult': self.lr_mult, 'decay_mult': self.decay_mult},
        ]
        return parameters


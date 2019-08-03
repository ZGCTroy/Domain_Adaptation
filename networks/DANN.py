from networks.Baseline import *


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
            in_features_size=self.base_model.features_output_size,
            lr_mult=self.lr_mult,
            decay_mult=self.decay_mult,
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

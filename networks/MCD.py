from networks.Baseline import *


class MCD(nn.Module):
    def __init__(self, n_classes, base_model, pretrained=True):
        super(MCD, self).__init__()

        self.n_classes = n_classes
        self.pretrained = pretrained
        self.base_model_name = base_model

        if base_model == 'ResNet50':
            self.in_features_size = 256
            self.Generator = ResNet50(n_classes=n_classes, pretrained=pretrained, bottleneck_dim=self.in_features_size)

            self.Classifier1 = get_large_classifier(
                in_features_size=self.in_features_size,
                n_classes=self.n_classes,
            )
            self.Classifier1.apply(init_weights)

            self.Classifier2 = get_large_classifier(
                in_features_size=self.in_features_size,
                n_classes=self.n_classes,
            )
            self.Classifier2.apply(init_weights)

            self.lr_mult = 10
            self.decay_mult = 2

        if base_model == 'DigitsStoM':
            self.Generator = DigitsStoM(n_classes=n_classes)
            self.in_features_size = 128

            self.Classifier1 = get_large_classifier(
                in_features_size=self.in_features_size,
                n_classes=self.n_classes
            )

            self.Classifier2 = get_large_classifier(
                in_features_size=self.in_features_size,
                n_classes=self.n_classes
            )

            self.lr_mult = 1
            self.decay_mult = 1

        if base_model == 'DigitsMU':
            self.Generator = DigitsMU(n_classes=n_classes)
            self.in_features_size = 1024

            self.Classifier1 = get_small_classifier(
                in_features_size=self.in_features_size,
                n_classes=self.n_classes
            )

            self.Classifier2 = get_small_classifier(
                in_features_size=self.in_features_size,
                n_classes=self.n_classes
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

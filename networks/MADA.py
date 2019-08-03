from networks.AdversarialNetwork import AdversarialNetwork
from networks.Baseline import *


class MADA(nn.Module):
    def __init__(self, n_classes, base_model, pretrained=True):
        super(MADA, self).__init__()

        self.n_classes = n_classes
        self.pretrained = pretrained

        if base_model == 'ResNet50':
            self.base_model = ResNet50(n_classes=n_classes, pretrained=pretrained, bottleneck_dim=256)
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
                    in_features_size=self.base_model.features_output_size,
                    lr_mult=self.lr_mult,
                    decay_mult=self.decay_mult,
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

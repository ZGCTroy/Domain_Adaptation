from networks.Baseline import *


class MT(nn.Module):
    def __init__(self, n_classes, base_model, pretrained=True):
        super(MT, self).__init__()

        self.n_classes = n_classes
        self.pretrained = pretrained

        if base_model == 'ResNet50':
            self.student = ResNet50(n_classes=n_classes, pretrained=pretrained, bottleneck_dim=2048,
                                    use_dropout=True)
            self.teacher = ResNet50(n_classes=n_classes, pretrained=pretrained, bottleneck_dim=2048,
                                    use_dropout=True)

        if base_model == 'DigitsStoM':
            self.student = DigitsStoM(n_classes=n_classes, use_dropout=True)
            self.teacher = DigitsStoM(n_classes=n_classes, use_dropout=True)

        if base_model == 'DigitsMU':
            self.student = DigitsMU(n_classes=n_classes, use_dropout=True)
            self.teacher = DigitsMU(n_classes=n_classes, use_dropout=True)

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

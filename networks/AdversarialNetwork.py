from torch import nn
from torch.autograd import Function

from networks.Baseline import init_weights


class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None


class AdversarialNetwork(nn.Module):
    def __init__(self, in_features_size, lr_mult=10, decay_mult=2, out_features_size=1, sigmoid=True):
        super(AdversarialNetwork, self).__init__()
        self.in_features_size = in_features_size
        self.lr_mult = lr_mult
        self.decay_mult = decay_mult
        self.out_features_size = out_features_size

        self.discriminator = nn.Sequential(
            nn.Linear(self.in_features_size, 1024),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(1024, self.out_features_size),
        )
        if sigmoid:
            self.discriminator.add_module('sigmoid',nn.Sigmoid())

        self.discriminator.apply(init_weights)

    def forward(self, x, alpha):
        x = ReverseLayerF.apply(x, alpha)

        y = self.discriminator(x)

        return y

    def get_parameters(self):
        parameters = [
            {"params": self.discriminator.parameters(), "lr_mult": self.lr_mult, 'decay_mult': self.decay_mult}
        ]
        return parameters

import torch
from torch import nn
import torch.nn.functional as F

# BaseLine network for "SVHN [3,32,32] -> MNIST [3,32,32]"
class BaselineStoM(nn.Module):
    def __init__(self, n_classes):
        super(BaselineStoM, self).__init__()

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

        self.fc4 = nn.Linear(128, n_classes)

    def forward(self, x):
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
        x = x.view(-1, 128)

        x = self.fc4(x)
        return x

# BaseLine network for "MNIST [1,28,28] <-> USPS[1,28,28]"
class BaselineMU(nn.Module):
    def __init__(self, n_classes):
        super(BaselineMU, self).__init__()
        self.conv1_1 = nn.Conv2d(1, 32, (5, 5))
        self.conv1_1_bn = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d((2, 2))

        self.conv2_1 = nn.Conv2d(32, 64, (3, 3))
        self.conv2_1_bn = nn.BatchNorm2d(64)
        self.conv2_2 = nn.Conv2d(64, 64, (3, 3))
        self.conv2_2_bn = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d((2, 2))

        self.drop1 = nn.Dropout()
        self.fc3 = nn.Linear(1024, 256)
        self.fc4 = nn.Linear(256, n_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1_1_bn(self.conv1_1(x))))
        x = F.relu(self.conv2_1_bn(self.conv2_1(x)))
        x = self.pool2(F.relu(self.conv2_2_bn(self.conv2_2(x))))
        x = x.view(-1, 1024)
        x = self.drop1(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

def test():
    print('hello')

if __name__ == '__main__':
    test()
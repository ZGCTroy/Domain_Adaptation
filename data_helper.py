from torchvision import datasets
import os
from transform import transform_for_Office
from torchvision import transforms
import numpy as np
import torch
from datasets import USPSDataset
from PIL import Image


def _check_exists(self):
    return os.path.exists(os.path.join(self.root, self.training_file)) and \
           os.path.exists(os.path.join(self.root, self.test_file))


def load_Office(root_dir, domain):
    root_dir = os.path.join(root_dir, domain)
    transform = transform_for_Office(resize_size=[192,192], crop_size=160)
    dataset = {
        'train': datasets.ImageFolder(
            root=root_dir,
            transform=transform['train']
        ),
        'test': datasets.ImageFolder(
            root=root_dir,
            transform=transform['test']
        )
    }
    return dataset


def load_SVHN(root_dir):
    T = {
        'train': transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.43777722, 0.4438628, 0.47288644], std=[0.19664814, 0.19963288, 0.19541258])
            # transforms.Normalize(mean=(0.5,), std=(0.5,))
        ]),
        'test': transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.4525405, 0.45260695, 0.46907398], std=[0.21789917, 0.22504489, 0.22678198])
            # transforms.Normalize(mean=(0.5,), std=(0.5,))
        ])
    }

    SVHN = {
        'train': datasets.SVHN(
            root=root_dir, split='train', download=True,
            transform=T['train']
        ),
        'test': datasets.SVHN(
            root=root_dir, split='test', download=True,
            transform=T['test']
        )
    }
    return SVHN


def load_USPS(root_dir):
    T = {
        'train': [
            transforms.ToPILImage()
        ],
        'test': [
            transforms.ToPILImage()
        ]
    }

    T['train'].append(transforms.Resize([28, 28],interpolation=Image.BILINEAR))
    T['test'].append(transforms.Resize([28, 28],interpolation=Image.BILINEAR))

    T['train'].append(transforms.ToTensor())
    T['test'].append(transforms.ToTensor())

    # if Gray_to_RGB:
    #     T['train'].append(transforms.Lambda(lambda x: x.expand([3, -1, -1])))
    #     T['test'].append(transforms.Lambda(lambda x: x.expand([3, -1, -1])))

    # T['train'].append(transforms.Normalize(mean=(0.25466308,), std=(0.35181096,)))
    # T['test'].append(transforms.Normalize(mean=(0.26791447,), std=(0.3605367,)))

    # T['train'].append(transforms.Normalize(mean=(0.5,), std=(0.5,)))
    # T['test'].append(transforms.Normalize(mean=(0.5,), std=(0.5,)))

    USPS = {
        'train': USPSDataset(
            root_dir=root_dir,
            train=True,
            transform=transforms.Compose(T['train']),
        ),
        'test': USPSDataset(
            root_dir=root_dir,
            train=False,
            transform=transforms.Compose(T['test']),
        ),
    }
    return USPS


def load_MNIST(root_dir, resize_size=28, Gray_to_RGB=False):
    T = {'train': [], 'test': []}

    if resize_size == 32:
        T['train'].append(transforms.Pad(padding=2, fill=0, padding_mode='constant'))
        T['test'].append(transforms.Pad(padding=2, fill=0, padding_mode='constant'))

    T['train'].append(transforms.ToTensor())
    T['test'].append(transforms.ToTensor())

    if Gray_to_RGB:
        T['train'].append(transforms.Lambda(lambda x: x.expand([3, -1, -1])))
        T['test'].append(transforms.Lambda(lambda x: x.expand([3, -1, -1])))

    # T['train'].append(transforms.Normalize(mean=(0.13065113,), std=(0.30767146,)))
    # T['test'].append(transforms.Normalize(mean=(0.13284597,), std=(0.30983892,)))
    #
    # T['train'].append(transforms.Normalize(mean=(0.5,), std=(0.5,)))
    # T['test'].append(transforms.Normalize(mean=(0.5,), std=(0.5,)))

    MNIST = {
        'train': datasets.MNIST(
            root=root_dir, train=True, download=True,
            transform=transforms.Compose(T['train'])
        ),
        'test': datasets.MNIST(
            root=root_dir, train=False, download=True,
            transform=transforms.Compose(T['test'])
        )
    }
    return MNIST


def cal_mean_and_std():
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    dset = load_USPS(root_dir='./data/Digits/USPS')

    # dset = load_MNIST(root_dir='./data/Digits/MNIST', Gray_to_RGB=True)
    data_loader = torch.utils.data.DataLoader(
        dset['test'],
        batch_size=128,
        shuffle=False,
        num_workers=0
    )

    data_mean = []  # Mean of the dataset
    data_std0 = []  # std of dataset
    data_std1 = []  # std with ddof = 1
    for i, data in enumerate(data_loader, 0):
        # shape (batch_size, 3, height, width)
        numpy_image = data[0].numpy()

        # shape (3,)
        batch_mean = np.mean(numpy_image, axis=(0, 2, 3))
        batch_std0 = np.std(numpy_image, axis=(0, 2, 3))
        batch_std1 = np.std(numpy_image, axis=(0, 2, 3), ddof=1)

        data_mean.append(batch_mean)
        data_std0.append(batch_std0)
        data_std1.append(batch_std1)

    # shape (num_iterations, 3) -> (mean across 0th axis) -> shape (3,)
    data_mean = np.array(data_mean).mean(axis=0)
    data_std0 = np.array(data_std0).mean(axis=0)
    data_std1 = np.array(data_std1).mean(axis=0)

    print(data_mean, data_std0, data_std1)


def main():
    cal_mean_and_std()

    # # USPS train [7291,1,16,16] test [2007,1,16,16]
    # USPS = load_USPS(root_dir='./data/Digits/USPS', resize_size=16)
    #
    # # SVHN train [73257,3,32,32] test [26032,3,32,32]
    # SVHN = load_SVHN(root_dir='./data/Digits/SVHN', resize_size=32)
    #
    # root_dir = './data/Office31'
    # # Office31 31 classes , Amazon 2817
    # Amazon = load_Amazon(os.path.join(root_dir, 'Amazon'), resize_size= 256, crop_size = 224)
    # print(Amazon)
    #
    # # Office31 31 classes , Dslr 498
    # Dslr = load_Dslr(os.path.join(root_dir, 'Dslr'), resize_size= 256, crop_size = 224)
    # print(Dslr)
    #
    # # Office31 31 classes , Webcam 795
    # Webcam = load_Webcam(os.path.join(root_dir, 'Webcam'), resize_size= 256, crop_size = 224)
    # print(Webcam)
    #
    # root_dir = './data/Office-Home'
    # # OfficeHome 65 classes , Art 2427 RGB
    # Art = load_Art(os.path.join(root_dir, 'Art'), resize_size= 256, crop_size = 224)
    # print(Art)
    #
    # # OfficeHome 65 classes , Clipart 4365 RGB
    # Clipart = load_Clipart(os.path.join(root_dir, 'Clipart'), resize_size= 256, crop_size = 224)
    # print(Clipart)
    #
    # # OfficeHome 65 classes , Product 4439 RGB
    # Product = load_Product(os.path.join(root_dir, 'Product'), resize_size= 256, crop_size = 224)
    # print(Product)
    #
    # # OfficeHome 65 classes , RealWorld 4357 RGB
    # RealWorld = load_RealWorld(os.path.join(root_dir, 'Real World'), resize_size= 256, crop_size = 224)
    # print(RealWorld)


if __name__ == '__main__':
    main()

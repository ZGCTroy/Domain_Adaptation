from torchvision import transforms, datasets
from torch.utils import data
from torchvision.datasets.utils import download_url
import os
import zipfile
from datasets import USPSDataset
from transform import define_specific_transform
import os
import gdown
import requests
import urllib

os.chdir(os.getcwd())

def _check_exists(self):
    return os.path.exists(os.path.join(self.root, self.training_file)) and \
           os.path.exists(os.path.join(self.root, self.test_file))

def load_Amazon(root_dir, image_size=224, Normalization = True):
    transform = define_specific_transform(resize=image_size, Normalization = Normalization, RGB=True)
    Amazon = {
        'train': datasets.ImageFolder(
            root=root_dir,
            transform=transform['train']
        ),
        'test': datasets.ImageFolder(
            root=root_dir,
            transform=transform['test']
        )
    }
    return Amazon

def load_Dslr(root_dir, image_size=224, Normalization = True):
    transform = define_specific_transform(resize=image_size, Normalization = Normalization, RGB=True)
    Dslr = {
        'train': datasets.ImageFolder(
            root=root_dir,
            transform=transform['train']
        ),
        'test': datasets.ImageFolder(
            root=root_dir,
            transform=transform['test']
        )
    }
    return Dslr

def load_Webcam(root_dir, image_size=224, Normalization = True):
    transform = define_specific_transform(resize=image_size, Normalization = Normalization, RGB=True)
    Webcam = {
        'train': datasets.ImageFolder(
            root=root_dir,
            transform=transform['train']
        ),
        'test': datasets.ImageFolder(
            root=root_dir,
            transform=transform['test']
        )
    }
    return Webcam

def load_Art(root_dir, image_size=224, Normalization = True):
    transform = define_specific_transform(resize=image_size, Normalization = Normalization, RGB=True)
    Art = {
        'train': datasets.ImageFolder(
            root=root_dir,
            transform=transform['train']
        ),
        'test': datasets.ImageFolder(
            root=root_dir,
            transform=transform['test']
        )
    }
    return Art

def load_Clipart(root_dir, image_size=224, Normalization = True):
    transform = define_specific_transform(resize=image_size, Normalization = Normalization, RGB=True)
    Clipart = {
        'train': datasets.ImageFolder(
            root=root_dir,
            transform=transform['train']
        ),
        'test': datasets.ImageFolder(
            root=root_dir,
            transform=transform['test']
        )
    }
    return Clipart

def load_Product(root_dir, image_size=224, Normalization = True):
    transform = define_specific_transform(resize=image_size, Normalization = Normalization, RGB=True)
    Product = {
        'train': datasets.ImageFolder(
            root=root_dir,
            transform=transform['train']
        ),
        'test': datasets.ImageFolder(
            root=root_dir,
            transform=transform['test']
        )
    }
    return Product

def load_RealWorld(root_dir, image_size=224, Normalization = True):
    transform = define_specific_transform(resize=image_size, Normalization = Normalization, RGB=True)
    RealWorld = {
        'train': datasets.ImageFolder(
            root=root_dir,
            transform=transform['train']
        ),
        'test': datasets.ImageFolder(
            root=root_dir,
            transform=transform['test']
        )
    }
    return RealWorld

def load_SVHN(root_dir, image_size =32, Normalization = True):
    transform = define_specific_transform(resize=image_size, Normalization = Normalization, RGB=True)
    SVHN = {
        'train': datasets.SVHN(
            root=root_dir, split='train', download=True,
            transform=transform['train']
        ),
        'test': datasets.SVHN(
            root=root_dir, split='test', download=True,
            transform=transform['test']
        )
    }
    return SVHN

def load_USPS(root_dir, image_size =16, Gray_to_RGB = False, Normalization = True):
    transform = define_specific_transform(resize=image_size, Gray_to_RGB=Gray_to_RGB, Normalization = Normalization, RGB=False)
    USPS = {
        'train': USPSDataset(
            root=root_dir, split='train', download=True,
            transform=transform['train']
        ),
        'test': USPSDataset(
            root=root_dir, split='test', download=True,
            transform=transform['test']
        )
    }
    return USPS

def load_MNIST(root_dir, image_size =28, Gray_to_RGB = False, Normalization = True):
    transform = define_specific_transform(resize=image_size, Gray_to_RGB=Gray_to_RGB, Normalization = Normalization, RGB=False)
    MNIST = {
        'train': datasets.MNIST(
            root=root_dir, train=True, download=True,
            transform=transform['train']
        ),
        'test': datasets.MNIST(
            root=root_dir, train=False, download=True,
            transform=transform['test']
        )
    }
    return MNIST


def main():
    # MNIST train [60000,1,28,28] test [10000,1,28,28]
    MNIST = load_MNIST(root_dir='./data/Digits/MNIST',image_size=[28,28])

    # USPS train [7291,1,16,16] test [2007,1,16,16]
    USPS = load_USPS(root_dir='./data/Digits/USPS', image_size=[16, 16])

    # SVHN train [73257,3,32,32] test [26032,3,32,32]
    SVHN = load_SVHN(root_dir='./data/Digits/SVHN', image_size=[32,32])

    root_dir = './data/Office31'
    # Office31 31 classes , Amazon 2817
    Amazon = load_Amazon(os.path.join(root_dir, 'Amazon'), image_size=[224, 224])
    print(Amazon)

    # Office31 31 classes , Dslr 498
    Dslr = load_Dslr(os.path.join(root_dir, 'Dslr'), image_size=[224, 224])
    print(Dslr)

    # Office31 31 classes , Webcam 795
    Webcam = load_Webcam(os.path.join(root_dir, 'Webcam'), image_size=[224, 224])
    print(Webcam)

    root_dir = './data/Office-Home'
    # OfficeHome 65 classes , Art 2427 RGB
    Art = load_Art(os.path.join(root_dir, 'Art'), image_size=[224, 224])
    print(Art)

    # OfficeHome 65 classes , Clipart 4365 RGB
    Clipart = load_Clipart(os.path.join(root_dir, 'Clipart'), image_size=[224, 224])
    print(Clipart)

    # OfficeHome 65 classes , Product 4439 RGB
    Product = load_Product(os.path.join(root_dir, 'Product'), image_size=[224, 224])
    print(Product)

    # OfficeHome 65 classes , RealWorld 4357 RGB
    RealWorld = load_RealWorld(os.path.join(root_dir, 'Real World'), image_size=[224, 224])
    print(RealWorld)

if __name__ == '__main__':
    main()

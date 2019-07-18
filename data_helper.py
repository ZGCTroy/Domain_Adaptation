from torchvision import transforms, datasets
from torch.utils import data
from torchvision.datasets.utils import download_url
import os
import zipfile
from datasets import USPSDataset
from transform import transform_for_Digits, transform_for_Office
import os
import gdown
import requests
import urllib

os.chdir(os.getcwd())

def _check_exists(self):
    return os.path.exists(os.path.join(self.root, self.training_file)) and \
           os.path.exists(os.path.join(self.root, self.test_file))

def load_Amazon(root_dir, resize_size = 256, crop_size = 224):
    transform = transform_for_Office(resize_size=resize_size, crop_size = crop_size)
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

def load_Dslr(root_dir, resize_size = 256, crop_size = 224):
    transform = transform_for_Office(resize_size=resize_size, crop_size=crop_size)
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

def load_Webcam(root_dir, resize_size = 256, crop_size = 224):
    transform = transform_for_Office(resize_size=resize_size, crop_size=crop_size)
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

def load_Art(root_dir, resize_size = 256, crop_size = 224):
    transform = transform_for_Office(resize_size=resize_size, crop_size=crop_size)
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

def load_Clipart(root_dir, resize_size = 256, crop_size = 224):
    transform = transform_for_Office(resize_size=resize_size, crop_size=crop_size)
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

def load_Product(root_dir, resize_size = 256, crop_size = 224):
    transform = transform_for_Office(resize_size=resize_size, crop_size=crop_size)
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

def load_RealWorld(root_dir, resize_size = 256, crop_size = 224):
    transform = transform_for_Office(resize_size=resize_size, crop_size=crop_size)
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

def load_SVHN(root_dir, resize_size = 32):
    transform = transform_for_Digits(resize_size=resize_size, Gray_to_RGB=False)
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

def load_USPS(root_dir, resize_size = 16, Gray_to_RGB = False):
    transform = transform_for_Digits(resize_size=resize_size, Gray_to_RGB=Gray_to_RGB)
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

def load_MNIST(root_dir, resize_size = 28, Gray_to_RGB = False):
    transform = transform_for_Digits(resize_size=resize_size, Gray_to_RGB=Gray_to_RGB)
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
    MNIST = load_MNIST(root_dir='./data/Digits/MNIST',resize_size=28)

    # USPS train [7291,1,16,16] test [2007,1,16,16]
    USPS = load_USPS(root_dir='./data/Digits/USPS', resize_size=16)

    # SVHN train [73257,3,32,32] test [26032,3,32,32]
    SVHN = load_SVHN(root_dir='./data/Digits/SVHN', resize_size=32)

    root_dir = './data/Office31'
    # Office31 31 classes , Amazon 2817
    Amazon = load_Amazon(os.path.join(root_dir, 'Amazon'), resize_size= 256, crop_size = 224)
    print(Amazon)

    # Office31 31 classes , Dslr 498
    Dslr = load_Dslr(os.path.join(root_dir, 'Dslr'), resize_size= 256, crop_size = 224)
    print(Dslr)

    # Office31 31 classes , Webcam 795
    Webcam = load_Webcam(os.path.join(root_dir, 'Webcam'), resize_size= 256, crop_size = 224)
    print(Webcam)

    root_dir = './data/Office-Home'
    # OfficeHome 65 classes , Art 2427 RGB
    Art = load_Art(os.path.join(root_dir, 'Art'), resize_size= 256, crop_size = 224)
    print(Art)

    # OfficeHome 65 classes , Clipart 4365 RGB
    Clipart = load_Clipart(os.path.join(root_dir, 'Clipart'), resize_size= 256, crop_size = 224)
    print(Clipart)

    # OfficeHome 65 classes , Product 4439 RGB
    Product = load_Product(os.path.join(root_dir, 'Product'), resize_size= 256, crop_size = 224)
    print(Product)

    # OfficeHome 65 classes , RealWorld 4357 RGB
    RealWorld = load_RealWorld(os.path.join(root_dir, 'Real World'), resize_size= 256, crop_size = 224)
    print(RealWorld)

if __name__ == '__main__':
    main()

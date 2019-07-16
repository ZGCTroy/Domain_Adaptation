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

def load_dataset(dataset_name, root_dir = './data'):
    if dataset_name not in ['Office-31','Office-Home','Digits']:
        print('No this dataset')
        return

    if not os.path.exists(root_dir):
        download_dataset(root_dir=root_dir,dataset_name=dataset_name)

    root_dir = os.path.join(root_dir, dataset_name)

    if dataset_name == 'Digits':
        return load_Digits(root_dir=root_dir)

    if dataset_name == 'Office-31':
        return load_Office31(root_dir=root_dir)

    if dataset_name == 'Office-Home':
        return load_OfficeHome(root_dir=root_dir)

def load_Office31(root_dir):
    Office31 = {}

    train_transform = transforms.Compose([
        transforms.Resize([224,224]),
        transforms.ToTensor()
    ])

    test_transform = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor()
    ])

    for domain_name in ['Amazon','Dslr','Webcam']:
        Office31[domain_name] = {
            'train': datasets.ImageFolder(root=os.path.join(root_dir, domain_name),transform = train_transform),
            'test': datasets.ImageFolder(root=os.path.join(root_dir, domain_name),transform = test_transform)
        }

    return Office31

def load_OfficeHome(root_dir):
    OfficeHome = {}

    train_transform = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor()
    ])

    test_transform = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor()
    ])

    for domain_name in ['Art', 'Clipart', 'Product','Real World']:
        OfficeHome[domain_name] = {
            'train': datasets.ImageFolder(root=os.path.join(root_dir, domain_name), transform=train_transform),
            'test': datasets.ImageFolder(root=os.path.join(root_dir, domain_name), transform=test_transform)
        }

    return OfficeHome

def load_Digits(root_dir):
    Digits = {
        'USPS': load_USPS(os.path.join(root_dir, 'USPS'), image_size=[16, 16], Gray_to_RGB=False),
        'MNIST' : load_MNIST(os.path.join(root_dir,'MNIST'), image_size =[28,28], Gray_to_RGB = False),
        'SVHN': load_SVHN(os.path.join(root_dir, 'SVHN'), image_size=[32, 32]),
    }

    return Digits

def load_SVHN(root_dir, image_size =[32,32]):
    transform = define_specific_transform(resize=image_size)
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

def load_USPS(root_dir, image_size =[16,16], Gray_to_RGB = False):
    transform = define_specific_transform(resize=image_size, Gray_to_RGB=Gray_to_RGB)
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

def load_MNIST(root_dir, image_size =[28,28], Gray_to_RGB = False):
    transform = define_specific_transform(resize=image_size, Gray_to_RGB=Gray_to_RGB)
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


if __name__ == '__main__':
    main()

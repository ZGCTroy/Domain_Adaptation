#import torch
#from torchvision import datasets
import os

def download_datasets(root_dir = '../data', MNIST=False,SVHN=Fasle):
    if MNIST:
        MNIST_dataset = datasets.MNIST(os.path.join(root_dir, 'MNIST'), train=True, transform=None, target_transform=None, download = true)
    if SVHN:
        SVHN_dataset = datasets.SVHN(os.path.join(root_dir, 'SVHN'), split='train', transform=None, target_transform=None, download = true)



def load_Office31():
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5],[0.2,0.2,0.2])
    ])

    Office31 = torchvision.datasets.ImageFolder(
        '../data/Office-31',
        transform
    )

    return Office31




def load_OfficeHome():
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5],[0.2,0.2,0.2])
    ])

    OfficeHome = torchvision.datasets.ImageFolder(
        '../data/Office-Home',
        transform
    )

    return OfficeHome

def load_MNIST():
    MNIST = {'train':'','test':''}

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.2, 0.2, 0.2])
    ])
    test_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.2, 0.2, 0.2])
    ])

    MNIST['train'] = datasets.MNIST(os.path.join(root_dir, 'MNIST'), train=True, transform=train_transform, download=True)
    MNIST['test'] = datasets.MNIST(os.path.join(root_dir, 'MNIST'), train=False, transform=test_transform, download=True)

    return MNIST


def load_SVHN():
    SVHN = {'train': '', 'test': ''}

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.2, 0.2, 0.2])
    ])
    test_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.2, 0.2, 0.2])
    ])

    SVHN['train'] = datasets.SVHN(os.path.join(root_dir, 'SVHN'), train=True, transform=train_transform,download=True)
    SVHN['test'] = datasets.SVHN(os.path.join(root_dir, 'SVHN'), train=False, transform=test_transform,download=True)

    return SVHN

def get_data_transforms():
    data_transforms = {}
    return data_transforms


def load_datasets(data_transforms):
    datasets = {
        'MNIST':{'train','test'},
        'SVHN':{'train','test'}
    }
    datasets['MNIST'] = load_MNIST()

    datasets['SVHN'] = load_SVHN()


    a, b = torch.utils.data.random_split(dataset = , lengths = [3,7])

    datasets['Office-31'] = torchvision.datasets.ImageFolder(
        os.path.join(root_dir, 'Office-31')
    )
    return datasets


def get_data_loaders(datasets, data_transforms):
    data_loaders = datasets['SVHN']



download_datasets(MNIST=True, SVHN=True)

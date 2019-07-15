from torchvision import transforms, datasets
from torch.utils import data
from torchvision.datasets.utils import download_url
import os
import zipfile



def _check_exists(self):
    return os.path.exists(os.path.join(self.root, self.training_file)) and \
           os.path.exists(os.path.join(self.root, self.test_file))

def download_dataset(dataset_name, root_dir):
    if os.path.exists(os.path.join(root_dir,dataset_name)):
        print('Dataset already exisits. Delete the Dataset folder if you would like to redownload it !')
        return

    urls = {
        'Digits2': 'http://statweb.stanford.edu/~tibs/ElemStatLearn/datasets/zip.train.gz'
    }

    file_name = urls[dataset_name].rpartition('/')[-1]
    download_url(urls[dataset_name], root=root_dir)

    z = zipfile.ZipFile('Digits.zip', 'r')
    z.extractall(path=root_dir)

def load_dataset(dataset_name, root_dir = '../data'):
    if dataset_name not in ['Office-31','Office-Home','Digits']:
        print('No this dataset')
        return

    root_dir = os.path.join(root_dir, dataset_name)
    if not os.path.exists(root_dir):
        os.mkdir(root_dir)
        download_url(url, root=root_dir,filename=filename, md5=None)

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
    MNIST = {'train':'','test':''}

    train_transform = [
        transforms.Resize([32, 32]),
        transforms.ToTensor()
    ]

    test_transform = [
        transforms.Resize([32,32]),
        transforms.ToTensor()
    ]

    Digits = {}

    Digits['SVHN'] = {
        'train': datasets.SVHN(
            root=os.path.join(root_dir, 'SVHN'), split='train', download=True,
            transform=transforms.Compose(train_transform)
        ),
        'test': datasets.SVHN(
            root=os.path.join(root_dir, 'SVHN'), split='test', download=True,
            transform=transforms.Compose(test_transform)
        )
    }

    train_transform.append(transforms.Lambda(lambda x: x.expand(3, -1, -1).clone()))
    test_transform.append(transforms.Lambda(lambda x: x.expand(3, -1, -1).clone()))

    Digits['MNIST'] = {
        'train': datasets.MNIST(
            root=os.path.join(root_dir, 'MNIST'),train=True,download = True,
            transform=transforms.Compose(train_transform)
        ),
        'test': datasets.MNIST(
            root=os.path.join(root_dir, 'MNIST'),train=False,download=True,
            transform=transforms.Compose(test_transform)
        )
    }

    Digits['USPS'] = {
        'train': USPS(
            root=os.path.join(root_dir, 'USPS'), split='train', download=True,
            transform=transforms.Compose(train_transform)
        ),
        'test': USPS(
            root=os.path.join(root_dir, 'USPS'), split='test', download=True,
            transform=transforms.Compose(test_transform)
        )
    }

    return Digits


def main():

    dataset = load_dataset(root_dir='../data',dataset_name='Office-31')
    for domain_name in dataset:
        domain = dataset[domain_name]
        data_loader = data.DataLoader(domain['train'], batch_size=4, shuffle=False, num_workers=0)
        a, b = iter(data_loader).next()
        print(domain_name)
        print(a.size(), b.size())
        print()

    dataset = load_dataset(root_dir='../data', dataset_name='Office-Home')
    for domain_name in dataset:
        domain = dataset[domain_name]
        data_loader = data.DataLoader(domain['train'], batch_size=4, shuffle=False, num_workers=0)
        a, b = iter(data_loader).next()
        print(domain_name)
        print(a.size(), b.size())
        print()

    dataset = load_dataset(root_dir='../data', dataset_name='Digits')
    for domain_name in dataset:
        domain = dataset[domain_name]
        data_loader = data.DataLoader(domain['train'], batch_size=4, shuffle=False, num_workers=0)
        a, b = iter(data_loader).next()
        print(domain_name)
        print(a.size(), b.size())
        print()

if __name__ == '__main__':
    main()
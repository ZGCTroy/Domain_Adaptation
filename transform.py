from torchvision import transforms

def transform_for_Digits(resize_size, Gray_to_RGB=False):
    T = {
        'train': [
            transforms.Resize(resize_size),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize((0.5,), (0.5,)),
            transforms.ToTensor()
        ],
        'test': [
            transforms.Resize(resize_size),
            transforms.Normalize((0.5,), (0.5,)),
            transforms.ToTensor()
        ]
    }

    if Gray_to_RGB:
        for phase in T:
            T[phase].append(transforms.Lambda(lambda x: x.expand([3, -1, -1]).clone()))

    for phase in T:
        T[phase] = transforms.Compose(T[phase])

    return T

def transform_for_Office(resize_size, crop_size):
    T = {
        'train': transforms.Compose([
            transforms.Resize(resize_size),
            transforms.RandomResizedCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(resize_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    }
    return T
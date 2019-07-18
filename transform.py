from torchvision import transforms

def define_specific_transform(resize, RGB = True, Gray_to_RGB=False, Normalization = True):
    transform = {
        'train': [
            transforms.RandomResizedCrop(resize),
            transforms.RandomHorizontalFlip(),
        ],
        'test': [
            transforms.RandomResizedCrop(resize),
        ]
    }

    for _ in transform:
        transform[_].append(transforms.ToTensor())

    if Gray_to_RGB:
        for _ in transform:
            transform[_].append(transforms.Lambda(lambda x: x.expand([3, -1, -1]).clone()))

    if Normalization:
        if RGB or Gray_to_RGB:
            for _ in transform:
                transform[_].append(transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]))
        else:
            for _ in transform:
                transform[_].append(transforms.Normalize(mean=(0.5), std=(0.2881)))

    for _ in transform:
        transform[_] = transforms.Compose(transform[_])

    return transform
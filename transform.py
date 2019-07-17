from torchvision import transforms

def define_specific_transform(resize, Gray_to_RGB=False,Normalization = True):
    transform = {
        'train': [
            transforms.Resize(resize),
        ],
        'test': [
            transforms.Resize(resize),
        ]
    }

    for _ in transform:
        transform[_].append(transforms.ToTensor())

    if Gray_to_RGB:
        for _ in transform:
            transform[_].append(transforms.Lambda(lambda x: x.expand(3, -1, -1).clone()))
            if Normalization:
                transform[_].append(transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.2881,0.2881,0.2881)))
    else:
        if Normalization:
            for _ in transform:
                transform[_].append(transforms.Normalize(mean=(0.5), std=(0.2881)))

    for _ in transform:
        transform[_] = transforms.Compose(transform[_])

    return transform
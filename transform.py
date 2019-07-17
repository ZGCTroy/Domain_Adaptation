from torchvision import transforms

def define_specific_transform(resize, Gray_to_RGB=False):
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

    for _ in transform:
        transform[_] = transforms.Compose(transform[_])

    return transform
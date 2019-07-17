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

    for type in transform:
        transform[type].append(transforms.ToTensor())
        transform[type].append(transforms.)

    if Gray_to_RGB:
        for type in transform:
            transform[type].append(transforms.Lambda(lambda x: x.expand(3, -1, -1).clone()))

    for type in transform:
        transform[type] = transforms.Compose(transform[type])

    return transform
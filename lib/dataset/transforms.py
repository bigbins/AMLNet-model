from torchvision import transforms
from PIL import ImageFilter


def build_transforms(config, is_train=True):
    func = []
    if config.DATASET.GRAY:
        func.append(lambda x: x.convert('L'))
        normalize = transforms.Normalize(mean=[0.5], std=[0.25])

    if is_train:
        func.append(transforms.RandomRotation(degrees=45, expand=False))
        func.append(transforms.RandomHorizontalFlip(p=0.5))
        func.append(transforms.RandomVerticalFlip(p=0.5))
        func.append(transforms.RandomAffine(degrees=0., scale=(0.9, 1.2)))
        func.append(transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3))

        func.append(transforms.Resize(config.MODEL.IMAGE_SIZE))
        func.append(transforms.ToTensor())
        func.append(transforms.RandomErasing(p=0.5, scale=(0.01, 0.1)))

    else:
        func.extend([
            transforms.Resize(config.MODEL.IMAGE_SIZE),
            transforms.ToTensor()
        ])

    return transforms.Compose(func)




from torchvision import transforms


CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)


def get_train_transforms():

    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Normalize(
            CIFAR100_MEAN,
            CIFAR100_STD
        )
    ])


def get_eval_transforms():

    return transforms.Normalize(
        CIFAR100_MEAN,
        CIFAR100_STD
    )
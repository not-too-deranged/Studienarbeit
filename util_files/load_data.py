import random

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from util_files import FilteredPlaces365


def split_data(dataset, split = 0.8):
    train_size = int(split * len(dataset))
    val_size = len(dataset) - train_size
    return torch.utils.data.random_split(dataset, [train_size, val_size])


def prepare_data(hparams):
    transform_train = transforms.Compose([
        transforms.Resize(hparams.INPUT_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(hparams.INPUT_SIZE, padding=hparams.PADDING),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet mean
            std=[0.229, 0.224, 0.225]  # ImageNet std
        ),
    ])

    transform_test = transforms.Compose([
        transforms.Resize((hparams.INPUT_SIZE, hparams.INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    match hparams.USED_DATASET:
        case "CIFAR-100":
            train_val_dataset = torchvision.datasets.CIFAR100(
                root="./data", train=True, transform=transform_train, download=True
            )

            test_dataset = torchvision.datasets.CIFAR100(
            root="./data", train=False, transform=transform_test, download=True
            )

            train_dataset, val_dataset = split_data(train_val_dataset)

        case "cats":
            # load cat dataset: https://github.com/Aml-Hassan-Abd-El-hamid/datasets

            train_val_dataset = torchvision.datasets.ImageFolder(
                root="./data/cat-dataset/train", transform=transform_train
            )

            test_dataset = torchvision.datasets.ImageFolder(
                root="./data/cat-dataset/test", transform=transform_test
            )

            train_dataset, val_dataset = split_data(train_val_dataset)

        case "places365":
            # Download Places365 dataset for third test case
            # """
            ALL_CLASSES = list(range(365))
            random.seed(1234)
            SELECTED_150 = sorted(random.sample(ALL_CLASSES, 150))

            print("loading data")
            # train dataset is used for train, val and split set, because original val is too small and test ist unlabelled
            train_val_test_dataset = torchvision.datasets.Places365(
                root="./data", split="train-standard", transform=transform_train,
                download=True, small=True
            )
            # dataset is reduced in size by only using 150 of the 365 classes
            train_val_test_dataset = FilteredPlaces365.start_filter(train_val_test_dataset, SELECTED_150)

            print("loaded dataset")

            train_dataset, val_test_dataset = split_data(train_val_test_dataset, split=0.6)
            val_dataset, test_dataset = split_data(val_test_dataset, split=0.5)

        case _:
            raise ValueError(f"Unknown dataset: {hparams.USED_DATASET}")

    # DataLoaders allow batching and shuffling
    # Set NUM_WORKERS=0 for compatibility with freeze support (e.g., PyInstaller executables)

    train_loader = DataLoader(train_dataset, batch_size=hparams.BATCH_SIZE, shuffle=True,
                              num_workers=hparams.NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=hparams.BATCH_SIZE, shuffle=False, num_workers=hparams.NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=hparams.BATCH_SIZE, shuffle=False,
                             num_workers=hparams.NUM_WORKERS)

    return train_loader, val_loader, test_loader
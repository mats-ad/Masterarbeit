import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
import os

def get_transforms(image_size, split):
    normalize = transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    
    if split == "train":
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.ToTensor(),
            normalize,
        ])
    
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ])
        
def make_food11_loaders(
    root: str = "./data/food11",
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 4,
    seed: int = 42,
    subsample_frac: float = 1.0,
):
    root = os.path.abspath(root)
    train_ds = datasets.ImageFolder(
        os.path.join(root, "training"),
        transform=get_transforms(image_size, "train"),
    )
    val_ds = datasets.ImageFolder(
        os.path.join(root, "validation"),
        transform=get_transforms(image_size, "eval"),
    )
    test_ds = datasets.ImageFolder(
        os.path.join(root, "evaluation"),
        transform=get_transforms(image_size, "eval"),
    )
    
    # subsample training set for dataset-scaling experiment (Forschungsfrage 1)
    if subsample_frac < 1.0:
        rng = np.random.default_rng(seed)
        n = len(train_ds)
        indices = rng.choice(n, size=int(n*subsample_frac), replace=False)
        train_ds = Subset(train_ds, indices)
        
    pin = torch.cuda.is_available()
    
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin,
        persistent_workers=(num_workers > 0),
    )
    
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin,
        persistent_workers=(num_workers > 0),
    )
    
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin,
        persistent_workers=(num_workers > 0),
    )
    
    return train_loader, val_loader, test_loader, len(train_ds.dataset.classes if subsample_frac < 1.0 else train_ds.classes)
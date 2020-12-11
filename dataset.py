
from torchvision import datasets, models, transforms
import torch
import os

CXR_data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomAffine(degrees=(-10,10), translate=(0.1, 0.1), scale=(0.85, 1.15)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=(0.9, 1.1)),  # , contrast=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

CXR_data_dir = 'CXR'
CXR_image_datasets = {x: datasets.ImageFolder(os.path.join(CXR_data_dir, x),
                                          CXR_data_transforms[x])
                  for x in ['train', 'test']}

CXR_dataloaders = {x: torch.utils.data.DataLoader(CXR_image_datasets[x], batch_size=16,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'test']}

CXR_dataset_sizes = {x: len(CXR_image_datasets[x]) for x in ['train', 'test']}
CXR_class_names = CXR_image_datasets['train'].classes

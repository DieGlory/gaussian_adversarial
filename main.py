import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler

import torchvision
from utils import train_model_with_FGSM
from FGSM import fgsm_attack

import numpy as np
import matplotlib.pyplot as plt

import time
import os
import copy
from torchvision import datasets, models, transforms
import torch
import os

from FGSM import fsgm_test
from dataset import CXR_dataloaders

if __name__ == "__main__":
    from dataset import CXR_image_datasets,CXR_dataloaders
    from utils import train_model

    CXR_data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomAffine(degrees=(-10, 10), translate=(0.1, 0.1), scale=(0.85, 1.15)),
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

    CXR_dataloaders = {x: torch.utils.data.DataLoader(CXR_image_datasets[x], batch_size=1,
                                                      shuffle=True, num_workers=4)
                       for x in ['train', 'test']}
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_ft = torch.load("data/FGSM_resnet50_CXR_19.pth")
    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=3, gamma=0.1)
    for i in [0,0.05,0.1,0.15,0.2,0.25]:
      fsgm_test(model_ft,device,CXR_dataloaders['test'],i)


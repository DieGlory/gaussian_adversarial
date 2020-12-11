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



if __name__ == "__main__":
    from dataset import CXR_image_datasets,CXR_dataloaders
    from utils import train_model

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_ft = torchvision.models.resnet50(pretrained=True)

    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 3)
    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=3, gamma=0.1)

    model_ft = train_model_with_FGSM(model_ft, criterion, optimizer_ft, exp_lr_scheduler,CXR_image_datasets,CXR_dataloaders,num_epochs=25)

    torch.save(model_ft, "data/FGSM_resnet50_CXR.pth")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision

import numpy as np

x = torch.ones((100,100),requires_grad=True)
y = torch.ones((100,100)) * 5
y.requires_grad=True

w1 = torch.ones((100,100),requires_grad=True)
w2 = torch.ones((100,100),requires_grad=True)

pred = x.mm(w1).mm(w2)


normal_loss = ((pred - y)**2).mean()* 0.7
normal_loss.backward()
print("x[0] :", x[0,0])
print("x.grad[0] :", x.grad[0,0])

print("w1[0] :", w1[0,0])
print("w1.grad[0] :", w1.grad[0,0])

orig_img = x.clone()
noisy_x = orig_img + 0.25 * x.grad.sign()
noisy_pred = noisy_x.mm(w1).mm(w2)
fgsm_loss = ((noisy_pred - y)**2).mean() * 0.3
optimizer=optim.SGD([w1,w2],lr=0.0001)

fgsm_loss.backward()
optimizer.zero_grad()


optimizer.step()
print("x[0] :", x[0,0])
print("x.grad[0] :", x.grad[0,0])

print("w1[0] :", w1[0,0])
print("w1.grad[0] :", w1.grad[0,0])





print()

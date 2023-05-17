import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split, Dataset
from torchvision.io import read_image
import pandas as pd
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.datasets import DatasetFolder

    

transform = transforms.Compose([transforms.Resize((64,64)),
                                transforms.ToTensor()],)

dataset = DatasetFolder("./img", loader=lambda x: Image.open(x), extensions=["jpeg","jpg"], transform=transform)


train_size = 20000
test_size = 3000
valid_size = len(dataset) - train_size - test_size


train_ds, valid_ds, test_ds = random_split(dataset, [train_size,valid_size, test_size])

print('Train ds len', len(train_ds))
print('valid ds len', len(valid_ds))
print('Test ds len', len(test_ds))

# @title Device Dataloader
import torch


# it is used to put the model into GPU device
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""

    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)


# it is used to check GPU device
def get_default_device():
    """Pick CPU if avalible, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


device = get_default_device()
device


from torch.utils.data.dataloader import DataLoader

batch_size = 128
train_loader = DataLoader(train_ds, batch_size, shuffle=True)
val_loader = DataLoader(valid_ds, batch_size)
test_loader = DataLoader(test_ds, batch_size)

train_dl = DeviceDataLoader(train_loader, device)
val_dl = DeviceDataLoader(val_loader, device)
test_dl = DeviceDataLoader(test_loader, device)


def training(train_dl, model, optimizer, util):
    model.train()
    batch_loss = []
    batch_acc = []
    for batch in train_dl:
        images, labels = batch
        outputs = model(images)
        loss = util.criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        batch_loss.append(loss.clone().detach().cpu())
        batch_acc.append(util.accuracy(outputs, labels))
    return torch.stack(batch_loss).mean(), torch.stack(batch_acc).mean()

def validating(val_dl, model, util):
    model.eval()
    batch_loss = []
    batch_acc = []
    with torch.no_grad():
        for batch in val_dl:
            images, labels = batch
            outputs = model(images)
            loss = util.criterion(outputs, labels)
            batch_loss.append(loss.clone().detach().cpu())
            batch_acc.append(util.accuracy(outputs, labels))
        return torch.stack(batch_loss).mean(), torch.stack(batch_acc).mean()


import torch.nn as nn
import torch.nn.functional as F


class calculate():
    def criterion(self, preds, labels):
        loss = F.cross_entropy(preds, labels)
        return loss

    def accuracy(self, outputs, labels):
        _, preds = torch.max(outputs, dim=1)
        return torch.tensor(torch.sum(preds == labels).item() / len(preds))



def print_all(epoch, lr, result):
    msg1 = 'Epoch [{}], '.format(epoch)
    msg2 = 'lr: {:.6f}, '.format(lr)
    msg3 = 'train_loss: {:.6f}, train_acc: {:.6f}, val_loss: {:.6f}, val_acc: {:.6f}'. \
        format(result['train_loss'][-1], result['train_accu'][-1],
               result['valid_loss'][-1], result['valid_accu'][-1])
    print(msg1 + msg2 + msg3)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def fit(epochs, max_lr, model, train_dl, val_dl, weight_decay=0, opt_func=torch.optim.Adam):
    tuned_optimizer = opt_func(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0,
                               amsgrad=False)
    sched = torch.optim.lr_scheduler.OneCycleLR(tuned_optimizer, max_lr, epochs=epochs, steps_per_epoch=1)

    util = calculate()
    lrs = []
    results = {}
    results['train_loss'] = []
    results['train_accu'] = []
    results['valid_loss'] = []
    results['valid_accu'] = []
    for epoch in range(epochs):
        train_loss, train_accu = training(train_dl, model, tuned_optimizer, util)
        valid_loss, valid_accu = validating(val_dl, model, util)
        sched.step()
        results['train_loss'].append(train_loss)
        results['train_accu'].append(train_accu)
        results['valid_loss'].append(valid_loss)
        results['valid_accu'].append(valid_accu)
        print_all(epoch, get_lr(tuned_optimizer), results)
    return results


def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
              nn.BatchNorm2d(out_channels),
              nn.ReLU(inplace=True)]
    if pool: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


"""class ZFNet(nn.Module):
     A implementation of the ZFNet architecture from the paper 'Visualizing and
        Understanding Convolutional Networks' by Zeiler and Fergus
    def __init__(self ):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 96, kernel_size=7, stride=2, padding=3, padding_mode='reflect')
        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, stride=2, padding=2, padding_mode='reflect')
        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
        self.fc6 = nn.Linear(9216,4096)
        self.fc7 = nn.Linear(4096,4096)
        self.fc8 = nn.Linear(4096,10)
        self.pool1 = nn.MaxPool2d(3,stride=2)
        self.pool2 = nn.MaxPool2d(3,stride=2)
        self.drop = nn.Dropout(0.5)
        self.drop = nn.Dropout(0.5)
        self.lrn = nn.LocalResponseNorm(size=5,alpha=10e-4,beta=0.75,k=2.0)

    def forward(self, x):
        x = self.lrn(self.pool1(F.relu(self.conv1(x))))
        x = self.lrn(self.pool2(F.relu(self.conv2(x))))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(F.relu(self.conv5(x)))
        x = x.view(-1,9216)
        x = F.relu(self.drop(self.fc6(x)))
        x = F.relu(self.drop(self.fc7(x)))
        x = self.fc8(x)
        return x"""


class aModel(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.conv1 = conv_block(in_channels, 64)
        self.conv2 = conv_block(64, 128, pool=True)
        self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))

        self.conv3 = conv_block(128, 256, pool=True)
        self.conv4 = conv_block(256, 512, pool=True)
        self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))
        self.classifier = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                        nn.Flatten(),
                                        nn.Linear(512, num_classes))
        self.feature = [0]*1

        self.deconv1 = nn.ConvTranspose2d(64, 3, kernel_size=3, padding=1,bias=False)
    def forward(self, xb):

        out = self.conv1(xb)

        self.feature[0] = out;

        out = self.conv2(out)
       
        out = self.res1(out) + out

        out = self.conv3(out)

        out = self.conv4(out)

        out = self.res2(out) + out

        out = self.classifier(out)

        return out

    def forward_deconv(self, x):
        x = nn.ReLU(x)
        x = self.deconv1(x)
        return x


model = to_device(aModel(3, 10), device)

#model = torch.load('model.pth')

model = to_device(model, device)
#model.train()

epoch =1
max_lr = 0.005
weight_decay = 1e-4
optimizer = torch.optim.Adam

import time
t1 = time.perf_counter()
results = fit(epoch, max_lr, model, train_dl, val_dl, weight_decay, opt_func=optimizer)
t2 = time.perf_counter()
print('Time taken for training: {:.2f}s'.format(t2-t1))
#print(model.feature)
x = model.forward_deconv( model.feature[0])
print(x)

print(model.feature)
#plt.imshow(model.feature[0].detach().numpy())
torch.save(model.state_dict(), 'model_params.pth')
torch.save(model, 'model.pth')

import matplotlib.pyplot as plt

train_acc = results["train_accu"]
val_acc = results['valid_accu']
plt.subplot(2,1,1)
plt.plot(train_acc,'-o', label='train_acc')
plt.plot(val_acc,'-x', label='val_acc')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend()
plt.title('Accuracy vs number of epochs')

train_loss = results['train_loss']
val_loss = results['valid_loss']
plt.subplot(2,1,2)
plt.plot(train_loss,'-o', label='train_loss')
plt.plot(val_loss,'-x', label='val_loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.title('loss vs number of epochs')

plt.tight_layout()
plt.show()

def predict(test_dl, model, util):
  model.eval()
  batch_pred_prob = []
  batch_pred_label = []
  batch_label = []
  with torch.no_grad():
    for batch in test_dl:
      images= batch
      outputs = model(images)
      pred_prob, pred_label = F.softmax(outputs, dim=1).max(1)
      batch_pred_prob.append(pred_prob.cpu())
      batch_pred_label.append(pred_label.cpu())
    return torch.cat(batch_pred_label).numpy()

def testing(test_dl, model, util):
  model.eval()
  batch_pred_prob = []
  batch_pred_label = []
  batch_label = []
  with torch.no_grad():
    for batch in test_dl:
      images, labels = batch
      outputs = model(images)
      pred_prob, pred_label = F.softmax(outputs, dim=1).max(1)
      batch_pred_prob.append(pred_prob.cpu())
      batch_pred_label.append(pred_label.cpu())
      batch_label.append(labels.cpu())
    return torch.cat(batch_label).numpy(), torch.cat(batch_pred_label).numpy()



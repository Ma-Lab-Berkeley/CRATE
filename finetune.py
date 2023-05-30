# -*- coding: utf-8 -*-
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import pandas as pd
import csv
import time

from utils import progress_bar
from data.randomaug import RandAugment
from model.rit import *
from model.vit import *
from data.dataset import *
# parsers
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate') # resnets.. 1e-3, Vit..1e-4
parser.add_argument('--opt', default="adamW")
parser.add_argument('--net', default='vit')
parser.add_argument('--bs', type=int, default=50)
parser.add_argument('--data', default="cifar10")
parser.add_argument('--classes',type=int, default=10)
parser.add_argument('--resume',type=int, default=0)
parser.add_argument('--randomaug',type=int, default=1)
parser.add_argument('--rand_aug_n',type=int, default=2)
parser.add_argument('--rand_aug_m',type=int, default=14)
parser.add_argument('--erase_prob',type=float, default=0.0)
parser.add_argument('--n_epochs', type=int, default='400')
parser.add_argument('--patch', default='4', type=int, help="patch for ViT")
parser.add_argument('--ckpt_dir', type=str, default='./',help='location for the pretrained rit weight')
parser.add_argument('--data_dir', type=str, default='./data',help='location for datasets')

args = parser.parse_args()

# take in args

use_amp = True
bs = args.bs

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
size=224
transform_train = transforms.Compose([
    transforms.RandomResizedCrop((size,size)),
    transforms.RandomHorizontalFlip(),
    transforms.RandAugment(args.rand_aug_n, args.rand_aug_m) if args.randomaug else transforms.TrivialAugmentWide(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=args.erase_prob),
])
print("size", size)
transform_test = transforms.Compose([
    transforms.Resize((size,size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
])

transet, testset = load_dataset(args.data, size=size, transform_train=transform_train, transform_test=transform_test, data_dir=args.data_dir)
trainloader = torch.utils.data.DataLoader(transet, batch_size=bs, shuffle=True, num_workers=8)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=8)
# Model factory..
print('==> Building model..')

if args.net == 'vit_tiny':
    net = vit_tiny_patch16(global_pool=True)
    net.load_state_dict(torch.load(args.ckpt_dir)['model'])
    net.head = nn.Linear(192, args.classes)
elif args.net == 'vit_small':
    net = vit_small_patch16(global_pool=True)
    net.load_state_dict(torch.load(args.ckpt_dir)['model'])
    net.head = nn.Linear(384, args.classes)
elif args.net == 'rit_tiny':
    net = rit_tiny()
    net.load_state_dict(torch.load(args.ckpt_dir)['model'])
    net.mlp_head = nn.Sequential(
        nn.LayerNorm(384),
        nn.Linear(384, args.classes)
    )
elif args.net == "rit_small":
    net = rit_small()
    net.load_state_dict(torch.load(args.ckpt_dir)['model'])
    net.mlp_head = nn.Sequential(
        nn.LayerNorm(576),
        nn.Linear(576, args.classes)
    )
elif args.net == "rit_base":
    net = rit_base()
    net.load_state_dict(torch.load(args.ckpt_dir)['model'])
    net.mlp_head = nn.Sequential(
        nn.LayerNorm(768),
        nn.Linear(768, args.classes)
    )
elif args.net == "rit_large":
    net = rit_large()
    net.load_state_dict(torch.load(args.ckpt_dir)['model'])
    net.mlp_head = nn.Sequential(
        nn.LayerNorm(1024),
        nn.Linear(1024, args.classes)
    )

# For Multi-GPU
if 'cuda' in device:
    print(device)
    print("using data parallel")
    net = torch.nn.DataParallel(net) # make parallel
    cudnn.benchmark = True
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/{}-ckpt.t7'.format(args.net))
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

# Loss is CE
criterion = nn.CrossEntropyLoss()

if args.opt == "adam":
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
elif args.opt == "sgd":
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9)  
elif args.opt == "adamW":
    print("using adamW")
    optimizer = optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.01, betas = (0.9, 0.999), eps = 1e-8)
# use cosine scheduling
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.n_epochs)

##### Training
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        # Train with amp
        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = net(inputs)
            loss = criterion(outputs, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    return train_loss/(batch_idx+1)

##### Validation
def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {"model": net.state_dict(),
              "optimizer": optimizer.state_dict(),
              "scaler": scaler.state_dict()}
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/'+args.net+'-{}-ckpt.t7'.format(args.patch))
        best_acc = acc
    
    os.makedirs("log", exist_ok=True)
    content = time.ctime() + ' ' + f'Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, val loss: {test_loss:.5f}, acc: {(acc):.5f}'
    print(content)
    with open(f'log/log_{args.net}_patch{args.patch}.txt', 'a') as appender:
        appender.write(content + "\n")
    return test_loss, acc

list_loss = []
list_acc = []

net.cuda()
for epoch in range(start_epoch, args.n_epochs):
    start = time.time()
    trainloss = train(epoch)
    val_loss, acc = test(epoch)
    
    scheduler.step(epoch-1) # step cosine scheduling
    
    list_loss.append(val_loss)
    list_acc.append(acc)

    # Write out csv..
    with open(f'log/log_{args.net}_patch{args.patch}.csv', 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(list_loss) 
        writer.writerow(list_acc) 
    print(list_loss)
    
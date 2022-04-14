#!/usr/bin/python3

import torch.nn as nn
from neuralnet_visualize.visualize import visualizer as nnviz

model = nn.Sequential(
    nn.Conv2d(3,32,kernel_size=3,stride=1, padding=(1,1)),
    nn.Conv2d(32,64,kernel_size=3,stride=1),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(kernel_size=2,stride=2),
    nn.Conv2d(64,128,kernel_size=3),
    nn.Conv2d(128,256,kernel_size=3),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(kernel_size=2,stride=2),
    nn.Conv2d(256,256,kernel_size=3),
    nn.Conv2d(256,512,kernel_size=3),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(kernel_size=2,stride=2),
    nn.Conv2d(512,512,kernel_size=2),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(kernel_size=2,stride=2),
    nn.Conv1d(512,256,kernel_size=3),
    nn.Flatten(),
    nn.Linear(230400,out_features=4096),
    nn.ReLU(inplace=True),
    nn.Linear(4096,512),
    nn.Linear(512,200),
    nn.Softmax(200)
)

net = nnviz()
net.from_pytorch(model)
net.visualize()
net.summarize()
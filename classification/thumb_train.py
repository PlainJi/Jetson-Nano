#!/usr/bin/env python3


import torch
import torchvision
import torchvision.transforms as transforms
from dataset import ImageClassificationDataset
import threading
import time
from utils import preprocess
import torch.nn.functional as F

BATCH_SIZE = 8  # number of pictures trained per time
epochs = 10     # train times
TASK = 'thumbs'
CATEGORIES = ['thumbs_up', 'thumbs_down']

TRANSFORMS = transforms.Compose([
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
dataset = ImageClassificationDataset(TASK + '_' + 'A', CATEGORIES, TRANSFORMS)
print("{} task with {} categories defined".format(TASK, CATEGORIES))


device = torch.device('cuda')
# ALEXNET
# model = torchvision.models.alexnet(pretrained=True)
# model.classifier[-1] = torch.nn.Linear(4096, len(dataset.categories))
# SQUEEZENET
# model = torchvision.models.squeezenet1_1(pretrained=True)
# model.classifier[1] = torch.nn.Conv2d(512, len(dataset.categories), kernel_size=1)
# model.num_classes = len(dataset.categories)
# RESNET 18
model = torchvision.models.resnet18(pretrained=True)
model.fc = torch.nn.Linear(512, len(dataset.categories))
# RESNET 34
# model = torchvision.models.resnet34(pretrained=True)
# model.fc = torch.nn.Linear(512, len(dataset.categories))

model = model.to(device)
optimizer = torch.optim.Adam(model.parameters())
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)


try:
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    model = model.train()

    while epochs > 0:
        print('epoch = {}'.format(epochs))
        i = 0
        sum_loss = 0.0
        error_count = 0.0
        for images, labels in iter(train_loader):
            # send data to device
            images = images.to(device)
            labels = labels.to(device)
            # zero gradients of parameters
            optimizer.zero_grad()
            # execute model to get outputs
            outputs = model(images)
            # compute loss
            loss = F.cross_entropy(outputs, labels)
            # run backpropogation to accumulate gradients
            loss.backward()
            # step optimizer to adjust parameters
            optimizer.step()

            # increment progress
            error_count += len(torch.nonzero(outputs.argmax(1) - labels).flatten())
            count = len(labels.flatten())
            i += count
            sum_loss += float(loss)
            print('progress: {} loss: {} accuracy: {}'.format(\
                    round(i/len(dataset), 2), sum_loss/i, 1.0-error_count/i))

        epochs -= 1
except Exception as e:
    print(e)

torch.save(model.state_dict(), './my_model_1.pth')
print('model saved!')










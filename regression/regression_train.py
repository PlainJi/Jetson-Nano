#!/usr/bin/env python3

# USB Camera (Logitech C270 webcam)
from jetcam.usb_camera import USBCamera
import torchvision.transforms as transforms
from dataset import XYDataset
import cv2
from jetcam.utils import bgr8_to_jpeg
import torch
import torchvision
import time
from utils import preprocess
import torch.nn.functional as F



#camera = USBCamera(width=224, height=224, capture_device=0) # confirm the capture_device number
#camera.running = True
#print("camera created")


TASK = 'face'
CATEGORIES = ['nose', 'left_eye', 'right_eye']

TRANSFORMS = transforms.Compose([
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

dataset = XYDataset(TASK + '_' + 'A', CATEGORIES, TRANSFORMS)
print("{} task with {} categories defined".format(TASK, CATEGORIES))


device = torch.device('cuda')
output_dim = 2 * len(dataset.categories)  # x, y coordinate for each category
# ALEXNET
# model = torchvision.models.alexnet(pretrained=True)
# model.classifier[-1] = torch.nn.Linear(4096, output_dim)
# SQUEEZENET
# model = torchvision.models.squeezenet1_1(pretrained=True)
# model.classifier[1] = torch.nn.Conv2d(512, output_dim, kernel_size=1)
# model.num_classes = len(dataset.categories)
# RESNET 18
model = torchvision.models.resnet18(pretrained=True)
model.fc = torch.nn.Linear(512, output_dim)
# RESNET 34
# model = torchvision.models.resnet34(pretrained=True)
# model.fc = torch.nn.Linear(512, output_dim)
model = model.to(device)
print("model configured!")


epochs = 10
BATCH_SIZE = 8
optimizer = torch.optim.Adam(model.parameters())
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

def train():
    global BATCH_SIZE, LEARNING_RATE, MOMENTUM, model, dataset, optimizer, eval_button, train_button, epochs

    try:
        train_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=True
        )
        model = model.train()

        while epochs > 0:
            i = 0
            sum_loss = 0.0
            error_count = 0.0
            for images, category_idx, xy in iter(train_loader):
                # send data to device
                images = images.to(device)
                xy = xy.to(device)
                optimizer.zero_grad()
                # execute model to get outputs
                outputs = model(images)
                # compute MSE loss over x, y coordinates for associated categories
                loss = 0.0
                for batch_idx, cat_idx in enumerate(list(category_idx.flatten())):
                    loss += torch.mean((outputs[batch_idx][2 * cat_idx:2 * cat_idx+2] - xy[batch_idx])**2)
                loss /= len(category_idx)
                # run backpropogation to accumulate gradients
                loss.backward()
                # step optimizer to adjust parameters
                optimizer.step()
                # increment progress
                count = len(category_idx.flatten())
                i += count
                sum_loss += float(loss)
                print('progress: {} loss:{}'.format(\
                        round(i/len(dataset), 2),
                        sum_loss/i,
                        ))

            epochs -= 1
    except Exception as e:
        print(e)

train()
torch.save(model.state_dict(), './my_xy_model.pth')
print('model saved!')

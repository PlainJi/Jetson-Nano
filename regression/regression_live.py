#!/usr/bin/env python3

from jetcam.usb_camera import USBCamera
import torchvision.transforms as transforms
import torch
import torchvision
from dataset import XYDataset
import time
from utils import preprocess
import torch.nn.functional as F
import cv2, os

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
model.load_state_dict(torch.load('./my_xy_model.pth'))
model = model.eval()
print("model configured!")

camera = USBCamera(width=224, height=224, capture_device=0) # confirm the capture_device number
print("camera created!")


while True:
    image = camera.read()
    preprocessed = preprocess(image)
    output = model(preprocessed).detach().cpu().numpy().flatten()
    for category in dataset.categories:
        category_index = dataset.categories.index(category)
        x = output[2 * category_index]
        y = output[2 * category_index + 1]
        print('{} {} {}'.format(dataset.categories[category_index], x, y))
        x = int(camera.width * (x/2.0+0.5))
        y = int(camera.height * (y/2.0+0.5))
        color = [0,0,0]
        color[category_index] = 255
        image = cv2.circle(image, (x, y), 8, tuple(color), 3)

    image_path = os.path.join(str(time.time())+'.jpg')
    cv2.imwrite(image_path, image)
    print('{} saved!'.format(image_path))







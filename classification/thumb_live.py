#!/usr/bin/env python3

import torchvision.transforms as transforms
from dataset import ImageClassificationDataset
import torch
import torchvision
import threading
from utils import preprocess
import torch.nn.functional as F
from jetcam.usb_camera import USBCamera


TASK = 'thumbs'
CATEGORIES = ['thumbs_up', 'thumbs_down']
DATASETS = 'A'

TRANSFORMS = transforms.Compose([
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

dataset = ImageClassificationDataset(TASK + '_' + DATASETS, CATEGORIES, TRANSFORMS)


device = torch.device('cuda')
model = torchvision.models.resnet18(pretrained=True)
model.fc = torch.nn.Linear(512, len(dataset.categories))
model = model.to(device)
model.load_state_dict(torch.load('./my_model.pth'))
model = model.eval()


camera = USBCamera(capture_device=0)
while True:
    image = camera.read()
    preprocessed = preprocess(image)
    output = model(preprocessed)
    output = F.softmax(output, dim=1).detach().cpu().numpy().flatten()
    print('%5.2f %5.2f' % (output[0], output[1]))






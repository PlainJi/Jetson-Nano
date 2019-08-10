#!/usr/bin/env python3

# USB Camera (Logitech C270 webcam)
from jetcam.usb_camera import USBCamera
camera = USBCamera(width=224, height=224, capture_device=0) # confirm the capture_device number

camera.running = True
print("camera created")





import torchvision.transforms as transforms
from dataset import ImageClassificationDataset

TASK = 'thumbs'
CATEGORIES = ['thumbs_up', 'thumbs_down']
DATASETS = ['A', 'B']

TRANSFORMS = transforms.Compose([
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

datasets = {}
for name in DATASETS:
    datasets[name] = ImageClassificationDataset(TASK + '_' + name, CATEGORIES, TRANSFORMS)

print("{} task with {} categories defined".format(TASK, CATEGORIES))






from jetcam.utils import bgr8_to_jpeg

# initialize active dataset
dataset = datasets[DATASETS[0]]


def save(path):
    print('save a pic in {}'.format(path))
    dataset.save_entry(camera.value, path)








import torch
import torchvision


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

def load_model(c):
    model.load_state_dict(torch.load('./my_model.pth'))

def save_model(c):
    torch.save(model.state_dict(), './my_model.pth')














import threading
import time
from utils import preprocess
import torch.nn.functional as F

def live(model, camera):
    global dataset
    while True:
        image = camera.value
        preprocessed = preprocess(image)
        output = model(preprocessed)
        output = F.softmax(output, dim=1).detach().cpu().numpy().flatten()
        print(output)

def start_live(change):
    execute_thread = threading.Thread(target=live, args=(model, camera))
    execute_thread.start()










BATCH_SIZE = 8
epochs = 10


optimizer = torch.optim.Adam(model.parameters())
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

def train_eval(is_training):
    global BATCH_SIZE, LEARNING_RATE, MOMENTUM, model, dataset, optimizer, epochs

    try:
        train_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=True
        )

        time.sleep(1)

        if is_training:
            model = model.train()
        else:
            model = model.eval()
        while epochs > 0:
            i = 0
            sum_loss = 0.0
            error_count = 0.0
            for images, labels in iter(train_loader):
                # send data to device
                images = images.to(device)
                labels = labels.to(device)

                if is_training:
                    # zero gradients of parameters
                    optimizer.zero_grad()

                # execute model to get outputs
                outputs = model(images)

                # compute loss
                loss = F.cross_entropy(outputs, labels)

                if is_training:
                    # run backpropogation to accumulate gradients
                    loss.backward()

                    # step optimizer to adjust parameters
                    optimizer.step()

                # increment progress
                error_count += len(torch.nonzero(outputs.argmax(1) - labels).flatten())
                count = len(labels.flatten())
                i += count
                sum_loss += float(loss)
                print('progress: {}'.format(i/len(dataset)))
                print('loss: {}'.format(sum_loss / i))
                print('accuracy: {}'.format(1.0 - error_count / i))

            if is_training:
                count -= 1
                print('count = {}'.format(count))
            else:
                break
    except e:
        pass
    model = model.eval()














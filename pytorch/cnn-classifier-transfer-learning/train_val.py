import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt 
import torchvision

from torchvision import models, datasets, transforms
import time,os,copy
from matplotlib import pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = 'cpu'

def imgshow(img):
    img = img / 2 + .5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.show()
    
    
batch = 32
data_url = './data/'
  

transform = transforms.Compose([transforms.RandomResizedCrop(224), 
                                transforms.ToTensor(),
                                transforms.Normalize((.5,.5,.5),(.5,.5,.5))])

datasets = {x : datasets.ImageFolder(data_url+x,transform=transform) 
           for x in ['train','test','val']}

classes = datasets['train'].classes

dataloaders = {x: torch.utils.data.DataLoader(datasets[x],
                                         batch_size=batch ,shuffle=True)
               for x in ['train','test','val'] }

dataset_sizes = {x: len(datasets[x]) for x in ['train', 'val']}
# images,labels = next(iter(dataloader))
# imgshow(torchvision.utils.make_grid(images))
# print(labels)

for f in ['train','test','val']:
    print(len(dataloaders[f]))


losses_train=[]
losses_val=[]
def train_model(model, criterion, optimizer, scheduler,num_epochs=2):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features 
model_ft.fc = nn.Linear(num_ftrs, 2) 
model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.01, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

model_trained = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler)

PATH='./pneumonia_net.pth'
torch.save(model_trained.state_dict(),PATH)



plt.plot(losses_train)
plt.plot(losses_val)




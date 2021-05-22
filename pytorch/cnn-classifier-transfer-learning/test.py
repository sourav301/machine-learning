import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt 
import torchvision

from torchvision import models, datasets, transforms
import time,os,copy

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = 'cpu'
def imgshow(img):
    img = img / 2 + .5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.show()
    
    
batch = 64
image_dataset = './data/test'
  

transform = transforms.Compose([transforms.RandomResizedCrop(224), 
                                transforms.ToTensor(),
                                transforms.Normalize((.5,.5,.5),(.5,.5,.5))])

dataset = datasets.ImageFolder(image_dataset,transform=transform)

classes = dataset.classes

dataloader = torch.utils.data.DataLoader(dataset,
                                         batch_size=batch ,shuffle=True)

images,labels = next(iter(dataloader))

imgshow(torchvision.utils.make_grid(images))
print(labels)



def train_model(model, criterion, optimizer, scheduler,num_epochs=2):
    since = time.time()
    best_model = copy.deepcopy(model.state_dict())
    best_acc = 0
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch,num_epochs-1))
        print('-'*10)
        
        for phase in ['train']:
            if phase=='train':
                model.train()
            else:
                model.eval()
                
            running_loss = 0
            running_corrects = 0
            
            inputs_count=0
            for inputs,labels in dataloader:
                inputs_count+=len(labels)
                inputs,labels = inputs.to(device),labels.to(device)
                optimizer.zero_grad()
                
                #Traninig
                with torch.set_grad_enabled(True):
                    outputs = model(inputs)
                    _,preds = torch.max(outputs,1)
                    loss = criterion(outputs,labels)
                    
                    loss.backward()
                    optimizer.step()
                    
                running_loss += loss.item()*inputs.size(0)
                running_corrects = torch.sum(preds==labels.data) 
                print("Training on ",inputs_count,"of",len(dataset),loss.item()*inputs.size(0))
            epoch_loss = running_loss/len(dataset)
            accuracy = running_corrects/len(dataset)
            
            print('{} Loss:{} Acc:{}'.format("Train",epoch_loss,accuracy))
                
    time_elapsed = time.time()-since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
            
    return model
        


model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features 
model_ft.fc = nn.Linear(num_ftrs, 2) 
model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

# model_trained = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler)

PATH='./pneumonia_net.pth'
# torch.save(model_trained.state_dict(),PATH)

model_ft.load_state_dict(torch.load(PATH))
 

total,correct=0,0
for data in dataloader:
    images,labels = data[0].to(device),data[1].to(device)
    outputs = model_ft(images)
    _,predicted = torch.max(outputs,1)
    total += labels.size(0)
    correct += (predicted==labels).sum().item()
    print('Accuracy of the network on the test images: %d %%' % (
    100 * correct / total))
    print(total)
    
    
    
print('Accuracy of the network on the test images: %d %%' % (
    100 * correct / total))




# Please note that the code used may not represent the code throughout each experiment
# Certain parameters were hard-coded and changed through the testing process

import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
from tqdm import tqdm # Displays a progress bar

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Dataset, Subset, DataLoader, random_split
from AugMax.augmax_modules.augmax import AugMaxDataset
from AugMax.dataloaders.fmnist import fmnist_dataloaders

import torchattacks


# Load the dataset and train, val, test splits
print("Loading datasets...")
# FASHION_transform = transforms.Compose([
#     transforms.ToTensor(), # Transform from [0,255] uint8 to [0,1] float
# ])
# FASHION_trainval = datasets.FashionMNIST('.', download=True, train=True, transform=FASHION_transform)
# FASHION_train = Subset(FASHION_trainval, range(25000))
# FASHION_val = Subset(FASHION_trainval, range(25000,30000))
# FASHION_test = datasets.FashionMNIST('.', download=True, train=False, transform=FASHION_transform)

# BATCH_SIZE=128
FASHION_train, FASHION_test = fmnist_dataloaders(
        ".", AugMax=None,
        mixture_width=3, mixture_depth=1, aug_severity=3)


FASHION_adversarial_test = Subset(FASHION_train, range(1000))
FASHION_val = Subset(FASHION_train, range(15000, 30000))
FASHION_train = Subset(FASHION_train, range(15000))

# Create dataloaders
# TODO: Experiment with different batch sizes
trainloader = DataLoader(FASHION_train, batch_size=128, shuffle=True)
valloader = DataLoader(FASHION_val, batch_size=15, shuffle=True)
testloader = DataLoader(FASHION_test, batch_size=10, shuffle=True)
atestloader = DataLoader(FASHION_adversarial_test, batch_size=1, shuffle=True)
print("Done!")

class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        # TODO: Design your own network, define layers here.
        # Here We provide a sample of two-layer fully-connected network from HW4 Part3.
        # Your solution, however, should contain convolutional layers.
        # Refer to PyTorch documentations of torch.nn to pick your layers. (https://pytorch.org/docs/stable/nn.html)
        # Some common Choices are: Linear, Conv2d, ReLU, MaxPool2d, AvgPool2d, Dropout
        # If you have many layers, consider using nn.Sequential() to simplify your code

        self.net = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.Dropout(.1)
        )

        self.fc = nn.Sequential(
            nn.Linear(120, 84), # from 28x28 input image to hidden layer of size 256
            nn.ReLU(),
            nn.Linear(84,10) # from hidden layer to 10 class scores
        )

    def forward(self,x):
        # TODO: Design your own network, implement forward pass here
        # x = x.view(-1,28*28) # Flatten each image in the batch
        x = self.net(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
        # x = self.fc1(x)
        # relu = nn.ReLU() # No need to define self.relu because it contains no parameters
        # x = relu(x)
        # x = self.fc2(x)
        # # The loss layer will be applied outside Network class
        # return x


class ResBlock(nn.Module):
    expansion = 4
    
    def __init__(self, in_channels, inter_channels, stride):
        super(ResBlock, self).__init__()
        conv1 = []
        conv1.append(nn.Conv2d(in_channels, inter_channels, 1, stride, 0))
        conv1.append(nn.BatchNorm2d(inter_channels))
        conv1.append(nn.ReLU())
        conv1.append(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1))
        conv1.append(nn.BatchNorm2d(inter_channels))
        conv1.append(nn.ReLU())
        conv1.append(nn.Conv2d(inter_channels, inter_channels * 2, 1, 1, 0))
        conv1.append(nn.BatchNorm2d(inter_channels * 2))
        self.conv1 = nn.Sequential(*conv1)
        
        conv2 = []
        conv2.append(nn.Conv2d(in_channels, inter_channels, 1, stride, 0))
        conv2.append(nn.BatchNorm2d(inter_channels))
        conv2.append(nn.ReLU())
        conv2.append(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1))
        conv2.append(nn.BatchNorm2d(inter_channels))
        conv2.append(nn.ReLU())
        conv2.append(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1))
        conv2.append(nn.BatchNorm2d(inter_channels))
        conv2.append(nn.ReLU())
        conv2.append(nn.Conv2d(inter_channels, inter_channels * 2, 1, 1, 0))
        conv2.append(nn.BatchNorm2d(inter_channels * 2))
        self.conv2 = nn.Sequential(*conv2)
        
        short = []
        if stride != 1 or in_channels != inter_channels * self.expansion:
            short.append(nn.Conv2d(in_channels, inter_channels * self.expansion, 1, stride, 0))
            short.append(nn.BatchNorm2d(inter_channels * self.expansion))
        self.short = nn.Sequential(*short)
        self.relu = nn.ReLU()
        
    def forward(self, x): 
        out = torch.cat([self.conv1(x), self.conv2(x)], 1)
        x = self.short(x)
        out = self.relu(out + x)
        return out

class ResNet(nn.Module):
    def __init__(self, block, block_list):
        super(ResNet, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1, 64, 3, 1, 1), 
                                   nn.BatchNorm2d(64), 
                                   nn.ReLU(), 
                                   nn.MaxPool2d(3, 2, 1))
        
        self.block1 = self.make_layers(block, 64, 64, block_list[0], 1)
        self.block2 = self.make_layers(block, 256, 128, block_list[1], 1)
        self.block3 = self.make_layers(block, 512, 256, block_list[2], 1)
        self.block4 = self.make_layers(block, 1024, 512, block_list[3], 2)
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.drop = nn.Dropout(0.6)
        self.fc = nn.Linear(2048, 10)
        
    def make_layers(self, block , in_channels, inter_channels, blocks_num, stride):
        layers = []
        layers.append(block(in_channels, inter_channels, stride))
        
        in_channels = inter_channels * block.expansion
        for _ in range(blocks_num - 1):
            layers.append(block(in_channels,inter_channels , 1))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.drop(x)
        x = self.fc(x)
        return x

device = "cuda" if torch.cuda.is_available() else "cpu" # Configure device
print(device)

# CHANGE MODEL HERE
#model = ResNet(ResBlock, [3,4,6,3]).to(device)
model = LeNet().to(device)
# model.load_state_dict(torch.load("./model.pth"))

# TODO: Define loss function 
criterion = nn.CrossEntropyLoss() # Specify the loss layer
# TODO: Modify the line below, experiment with different optimizers and parameters (such as learning rate)
# optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4) # Specify optimizer and assign trainable parameters to it, weight_decay is L2 regularization strength
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4) 
num_epoch = 20 # TODO: Choose an appropriate number of training epochs

def train(model, loader, loader2, num_epoch = num_epoch): # Train the model
    print("Start training...")
    train_losses = []
    val_losses = []
    model.train() # Set the model to training mode
# model.load_state_dict(torch.load("./model.pth"))
    for i in range(num_epoch):
        train_step_loss = []
        for batch, label in tqdm(loader):
            batch = batch.to(device)
            label = label.to(device)
            optimizer.zero_grad() # Clear gradients from the previous iteration
            pred = model(batch) # This will call Network.forward() that you implement
            loss = criterion(pred, label) # Calculate the loss
            
            loss_item = loss.item()
            train_step_loss.append(loss_item)

            loss.backward() # Backprop gradients to all tensors in the network
            optimizer.step() # Update trainable weights
        
        mean = np.mean(train_step_loss)
        train_losses.append(mean)
        print("Epoch {} loss:{}".format(i+1,mean)) # Print the average loss for this epoch
        
        # val_step_loss = []
        # model.eval()
        # correct = 0
        # for batch, label in tqdm(loader2): 
        #     batch = batch.to(device)
        #     label = label.to(device)
        #     optimizer.zero_grad()
        #     pred = model(batch)
        #     loss = criterion(pred, label)

        #     val_step_loss.append(loss.item())
        #     correct += (torch.argmax(pred,dim=1)==label).sum().item()
        # val_losses.append(np.mean(val_step_loss))
        # acc = correct/len(loader.dataset)
        # print("Evaluation accuracy: {}".format(acc))

    print("Done!")

    return train_losses, val_losses

def adv_train(model, loader, test_loader, num_epoch = num_epoch): # Train the model
    print("Start training...")
    train_losses = []
    model.train() # Set the model to training mode
    
    total = correct = total_adv = correct_adv = step = 0
    atk = torchattacks.PGD(model)

    for i in range(num_epoch):
        train_step_loss = []
        for batch, label in tqdm(loader):
            batch = batch.to(device)
            label = label.to(device)
            optimizer.zero_grad() # Clear gradients from the previous iteration
            pred = model(batch) # This will call Network.forward() that you implement
            loss = criterion(pred, label) # Calculate the loss
            
            loss_item = loss.item()
            train_step_loss.append(loss_item)

            loss.backward() # Backprop gradients to all tensors in the network
            optimizer.step() # Update trainable weights
            
            #Adversarial training
            
            # adv_samples = linf_pgd(model, batch, label, 25/255, 2/255, 2)
            adv_samples = atk(batch, label)
            xs, ys = Variable(adv_samples), Variable(label)

            optimizer.zero_grad()
            pred = model(xs)
            loss = criterion(pred, ys)
            loss.backward()

            optimizer.step()

        
        mean = np.mean(train_step_loss)
        train_losses.append(mean)
        print("Epoch {} loss:{}".format(i+1,mean)) # Print the average loss for this epoch
        
    print("Done!")

    return train_losses


def val_evaluate(model, loader): # Evaluate accuracy on validation / test set
    losses = []
    model.eval() # Set the model to evaluation mode
    correct = 0
    for i in range(num_epoch):
        running_loss = []
        for batch, label in tqdm(loader):
            batch = batch.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            pred = model(batch)
            loss = criterion(pred, label)

            running_loss.append(loss.item())
            correct += (torch.argmax(pred,dim=1)==label).sum().item()
        losses.append(np.mean(running_loss))


    acc = correct/len(loader.dataset)
    print("Evaluation accuracy: {}".format(acc))

    return acc, losses

def evaluate(model, loader): # Evaluate accuracy on validation / test set
    model.eval() # Set the model to evaluation mode
    correct = 0
    with torch.no_grad(): # Do not calculate grident to speed up computation
        for batch, label in tqdm(loader):
            batch = batch.to(device)
            label = label.to(device)
            pred = model(batch)
            correct += (torch.argmax(pred,dim=1)==label).sum().item()
    acc = correct/len(loader.dataset)
    print("Evaluation accuracy: {}".format(acc))
    return acc


def conduct_training():
    training_losses, validation_losses = train(model, trainloader, valloader, num_epoch)

    print("Evaluate on test set")
    evaluate(model, testloader)
    torch.save(model.state_dict(), './model_lenet_augmix.pth')


    # plt.figure(figsize=(10,5))
    # plt.title("Training and Validation Loss across Iterations")
    # plt.plot(validation_losses,label="validation")
    # plt.plot(training_losses,label="training")
    # plt.xlabel("Iterations")
    # plt.ylabel("Loss")
    # plt.legend()
    # plt.show()


def conduct_adversarial_training():
    training_losses = adv_train(model, trainloader, testloader, num_epoch)

    print("Evaluate on test set")
    evaluate(model, testloader)
    torch.save(model.state_dict(), './model_lenet_augmix_adtrain.pth')


def linf_pgd(model, batch, label, e, a, steps) :
    batch = batch.to(device)
    label = label.to(device)
        
    original = batch.data
        
    for _ in range(steps) :    
        batch.requires_grad = True
        pred = model(batch)

        model.zero_grad()
        loss = criterion(pred, label)
        loss.backward()

        first = torch.clamp( (batch + a*batch.grad.sign()) - original, min=-e, max=e)
        batch = torch.clamp(original + first, min=0, max=1).detach_()
    return batch

def l2_pgd(model, batch, label, e, a, steps):
    batch = batch.to(device)
    label = label.to(device)

    for _ in range(steps):
        batch.requires_grad = True
        pred = model(batch)

        model.zero_grad()
        loss = criterion(pred, label)
        loss.backward()

        first = batch + a*batch.grad.sign()

        delta = first - batch
        delta_norms = torch.norm(delta.view(75, -1), p=2,dim=1)
        factor = e/delta_norms
        factor = torch.min(factor, torch.ones_like(delta_norms))

        delta = delta * factor.view(-1,1,1,1)
        batch = torch.clamp(batch + delta, min=0, max=1).detach()        



def pgd_attack():
    model.load_state_dict(torch.load("./model_lenet_augmix_adtrain.pth"))
    model.eval()
    
    for steps in [10]:
        print(steps)
        correct = 0
        count = 0
        
        atk = torchattacks.PGD(model) #change attack norm method here
        for batch, label in tqdm(atestloader):
            
            #batch = linf_pgd(model, batch, label, 25/255, 2/255, steps)
            batch = atk(batch, label)
            label = label.to(device)

            pred = model(batch)

            correct += (torch.max(pred.data,1)[1] == label).sum()
            count += 1
            
        incorrect = len(atestloader)-correct
        final_acc = correct/float(len(atestloader))
        ASR = 1 - final_acc

        print("Test ASR = {} / {} = {}".format(incorrect, len(atestloader), ASR))
        print('Accuracy: %f', final_acc)

if __name__ == "__main__":
    pgd_attack() # Change function here for training, adv training, or pgd attack


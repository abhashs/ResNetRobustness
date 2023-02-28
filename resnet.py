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

FASHION_val = Subset(FASHION_train, range(25000,30000))
FASHION_train = Subset(FASHION_train, range(25000))

# Create dataloaders
# TODO: Experiment with different batch sizes
trainloader = DataLoader(FASHION_train, batch_size=75, shuffle=True)
valloader = DataLoader(FASHION_val, batch_size=15, shuffle=True)
testloader = DataLoader(FASHION_test, batch_size=10, shuffle=True)
atestloader = DataLoader(FASHION_test, batch_size=1, shuffle=True)
print("Done!")

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
model = ResNet(ResBlock, [3,4,6,3]).to(device)
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
        
        val_step_loss = []
        model.eval()
        correct = 0
        for batch, label in tqdm(loader2): 
            batch = batch.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            pred = model(batch)
            loss = criterion(pred, label)

            val_step_loss.append(loss.item())
            correct += (torch.argmax(pred,dim=1)==label).sum().item()
        val_losses.append(np.mean(val_step_loss))
        acc = correct/len(loader.dataset)
        print("Evaluation accuracy: {}".format(acc))

    print("Done!")

    return train_losses, val_losses

def adv_train(model, loader, test_loader, num_epoch = num_epoch): # Train the model
    print("Start training...")
    train_losses = []
    model.train() # Set the model to training mode
    
    total = correct = total_adv = correct_adv = step = 0

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
            adv_samples = linf_pgd(model, batch, label, 25/255, 2/255, 2)
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


def part1():
    training_losses, validation_losses = train(model, trainloader, valloader, num_epoch)

    print("Evaluate on test set")
    evaluate(model, testloader)
    torch.save(model.state_dict(), './model.pth')


    plt.figure(figsize=(10,5))
    plt.title("Training and Validation Loss across Iterations")
    plt.plot(validation_losses,label="validation")
    plt.plot(training_losses,label="training")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


def conduct_adversarial_training():
    training_losses = adv_train(model, trainloader, testloader, num_epoch)

    print("Evaluate on test set")
    evaluate(model, testloader)
    torch.save(model.state_dict(), './model.pth')


def fgsm(image, e, data_grad):
    return image + e * data_grad.sign()


def part2_fgsm():
    model.load_state_dict(torch.load("./model.pth"))
    model.eval() # Set the model to evaluation mode

    correct = 0
    p = 25/255

    for batch, label in tqdm(atestloader):
        batch.requires_grad = True

        batch = batch.to(device)
        label = label.to(device)

        pred = model(batch)
        # base_pred = torch.argmax(pred, keepdim=True)[1]
        base_pred = pred.max(1, keepdim=True)[1]

        if base_pred.item() == label.item():
            # loss = criterion(pred, label)
            loss = F.nll_loss(pred, label)
            model.zero_grad()

            loss.backward()
            data_grad = batch.grad.data

            perturbed_data = fgsm(batch, p, data_grad)
            pred = model(perturbed_data)

            final_pred = pred.max(1, keepdim=True)[1]
            # final_pred = torch.argmax(pred, keepdim=True)[1]
            if final_pred.item() == label.item():
                correct += 1

    final_acc = correct/float(len(atestloader))
    # print("Attack Success Rate = {}".format(p, correct, len(atestloader), 1 - final_acc))
    print("Attack Success Rate = {}".format(1 - final_acc))

    return final_acc


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


def part2_pgd():
    model.load_state_dict(torch.load("./model_lenet.pth"))
    model.eval()
    
    for steps in [10]:
        correct = 0
        count = 0

        for batch, label in tqdm(atestloader):
            batch = l2_pgd(model, batch, label, 25/255, 2/255, steps)
            label = label.to(device)

            pred = model(batch)

            correct += (torch.max(pred.data,1)[1] == label).sum()
            count += 1
            
        print('Accuracy: %f', (100 * float(correct) / count))

def l2pgd_attack():
    model.load_state_dict(torch.load("./model_lenet.pth"))
    attack = torchattacks.PGDL2(model, eps=25/255, alpha=2/255, steps=10)

    correct = 0
    count = 0

    for batch, label in tqdm(atestloader):
        # batch = l2_pgd(model, batch, label, 25/255, 2/255, steps)
        batch = attack(batch, label)
        label = label.to(device)

        pred = model(batch)

        correct += (torch.max(pred.data,1)[1] == label).sum()
        count += 1

    print('Accuracy: %f', (100 * float(correct) / count))



if __name__ == "__main__":
    l2pgd_attack()


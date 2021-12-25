from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import models, transforms
import torchvision.datasets as dset
import copy

# device check
device = torch.device("cpu")
print('running device : {}'.format(device))

# Training settings
batch_size = 8

train_set = dset.ImageFolder(root='../dataset/train', transform=transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
]))

validation_set = dset.ImageFolder(root='../dataset/val', transform=transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()]))

test_set = dset.ImageFolder(root='../dataset/test', transform=transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()]))

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                           batch_size=batch_size,
                                           shuffle=True)

validation_loader = torch.utils.data.DataLoader(dataset=validation_set,
                                                batch_size=batch_size,
                                                shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_set,
                                          batch_size=batch_size,
                                          shuffle=True)

# Load the pretrained model from pytorch
vgg16 = models.vgg16_bn()
vgg16.load_state_dict(torch.load("../input/vgg16_bn.pth"))

# Freeze training for all layers
for param in vgg16.features.parameters():
    param.require_grad = False

# Newly created modules have require_grad=True by default
num_features = vgg16.classifier[6].in_features
features = list(vgg16.classifier.children())[:-1]  # Remove last layer
features.extend([nn.Linear(num_features, 4)]) # Add our layer with 4 outputs
vgg16.classifier = nn.Sequential(*features)  # Replace the model classifier
print(vgg16)

vgg16.to(device)
criterion = nn.CrossEntropyLoss()
optimizer_ft = optim.SGD(vgg16.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)


def eval_model(vgg, criterion):
    loss_test = 0
    acc_test = 0

    test_batches = len(test_loader)
    print("Evaluating model")
    print('-' * 30)

    for i, data in enumerate(test_loader):
        if i % 5 == 0:
            print("\rTest batch {}/{}".format(i, test_batches), end='')

        vgg.train(False)
        vgg.eval()
        inputs, labels = data

        with torch.no_grad():
            inputs, labels = Variable(inputs), Variable(labels)
        outputs = vgg(inputs)

        _, preds = torch.max(outputs.data, 1)
        loss = criterion(outputs, labels)

        loss_test += loss.data
        acc_test += torch.sum(preds == labels.data)

        del inputs, labels, outputs, preds
        torch.cuda.empty_cache()

    avg_loss = loss_test / len(test_set)
    avg_acc = acc_test / len(test_set)

    print()
    print('-' * 30)
    print("Test results")
    print("Avg loss (test): {:.4f}".format(avg_loss))
    print("Avg acc (test): {:.4f}".format(avg_acc))
    print('-' * 30)


def train_model(vgg, criterion, optimizer, scheduler, num_epochs=2):
    best_model_wts = copy.deepcopy(vgg.state_dict())
    best_acc = 0.0

    train_batches = len(train_loader)
    val_batches = len(validation_loader)

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch + 1, num_epochs))
        print('-' * 30)

        loss_train = 0
        loss_val = 0
        acc_train = 0
        acc_val = 0

        vgg.train(True)

        for i, data in enumerate(train_loader):
            if i % 5 == 0:
                print("\rTraining batch {}/{}".format(i, train_batches), end='')

            inputs, labels = data
            with torch.no_grad():
                inputs, labels = Variable(inputs), Variable(labels)
            optimizer.zero_grad()

            outputs = vgg(inputs)
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            loss_train += loss.data
            acc_train += torch.sum(preds == labels.data)

            del inputs, labels, outputs, preds
            torch.cuda.empty_cache()

        avg_loss = loss_train / len(train_set)
        avg_acc = acc_train / len(train_set)

        vgg.train(False)
        vgg.eval()

        print()
        print("Epoch {} result: ".format(epoch))
        print("Avg loss (train): {:.4f}".format(avg_loss))
        print("Avg acc (train): {:.4f}".format(avg_acc))
        print('-' * 30)
        print()

        for i, data in enumerate(validation_loader):
            if i % 5 == 0:
                print("\rValidation batch {}/{}".format(i, val_batches), end='', flush=True)

            inputs, labels = data

            with torch.no_grad():
                inputs, labels = Variable(inputs), Variable(labels)

            optimizer.zero_grad()
            outputs = vgg(inputs)

            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            loss_val += loss.data
            acc_val += torch.sum(preds == labels.data)

            del inputs, labels, outputs, preds
            torch.cuda.empty_cache()

        avg_loss_val = loss_val / len(validation_set)
        avg_acc_val = acc_val / len(validation_set)

        print()
        print("Epoch {} result: ".format(epoch))
        print("Avg loss (val): {:.4f}".format(avg_loss_val))
        print("Avg acc (val): {:.4f}".format(avg_acc_val))
        print('-' * 30)
        print()

        if avg_acc_val > best_acc:
            best_acc = avg_acc_val
            best_model_wts = copy.deepcopy(vgg.state_dict())

    print("Best training acc: {:.4f}".format(best_acc))

    vgg.load_state_dict(best_model_wts)
    return vgg


print("Test before training")
eval_model(vgg16, criterion)
vgg16 = train_model(vgg16, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=2)
print("Test after training")
eval_model(vgg16, criterion)

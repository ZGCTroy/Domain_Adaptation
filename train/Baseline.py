from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
import matplotlib.pyplot as plt
import time
import copy

from models.baseline import BaselineMU

plt.ion()   # interactive mode

from data_helper import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

def train(model, data_loader, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model = copy.deepcopy(model.state_dict())
    best_test_acc = 0.0

    train_data_num = len(data_loader['source']['train'])
    test_data_num = len(data_loader['target']['test'])

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # train
        train_loss = 0
        train_corrects = 0
        scheduler.step()
        model.train()
        for inputs, labels in data_loader['source']['train']:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            print(inputs.size(),outputs.size(),labels.size())
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            train_corrects += torch.sum(preds == labels.data)

        # test
        test_loss = 0
        test_corrects = 0
        model.eval()
        for inputs, labels in data_loader['target']['test']:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            test_loss = criterion(outputs, labels)
            test_loss.backward()

            test_loss += loss.item() * inputs.size(0)
            test_corrects += torch.sum(preds == labels.data)

        epoch_train_loss = train_loss / train_data_num
        epoch_train_acc = train_corrects.double() / train_data_num
        epoch_test_loss = test_loss / test_data_num
        epoch_test_acc = test_corrects.double() / test_data_num

        print('Train Loss: {:.4f} Train Acc: {:.4f}'.format(epoch_train_loss, epoch_train_acc))
        print('Test Loss: {:.4f} Test Acc: {:.4f}'.format(epoch_test_loss, epoch_test_acc))

        # deep copy the model
        if epoch_test_acc > best_test_acc:
            best_test_acc = epoch_test_acc
            best_model = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since

    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best test Acc: {:4f}'.format(best_test_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def main():
    # MNIST = load_MNIST(root_dir='/gdrive/My Drive/BaseLine/data/MNIST', image_size=[32, 32], Gray_to_RGB=True)
    # USPS = load_USPS(root_dir='/gdrive/My Drive/BaseLine/data/USPS', image_size=[32, 32], Gray_to_RGB=True)
    MNIST = load_MNIST(root_dir='./data/Digits/MNIST', image_size=[28, 28], Gray_to_RGB=False)
    USPS = load_USPS(root_dir='./data/Digits/USPS', image_size=[28, 28], Gray_to_RGB=False)

    source_data = MNIST
    target_data = USPS

    source_data_loader = {
        'train': torch.utils.data.DataLoader(source_data['test'], batch_size=256, shuffle=True, num_workers=4),
        'test': torch.utils.data.DataLoader(source_data['test'], batch_size=256, shuffle=False, num_workers=4)
    }
    target_data_loader = {
        'train': torch.utils.data.DataLoader(target_data['train'], batch_size=256, shuffle=True, num_workers=4),
        'test': torch.utils.data.DataLoader(target_data['test'], batch_size=256, shuffle=False, num_workers=4)
    }

    model = BaselineMU(n_classes = 10)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    model = train(
        model = model,
        data_loader = {
            'source':source_data_loader,
            'target':target_data_loader
        },
        criterion = nn.CrossEntropyLoss(),
        optimizer = optimizer,
        scheduler = exp_lr_scheduler,
        num_epochs=25
    )

if __name__ == '__main__':
    main()
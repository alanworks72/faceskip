import os
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from src.dataloader_fer import loadBatches
from src.model_fer import loadModel
from utils import checkDir


def train(model, train_batches, valid_batches, epochs, opt, cri, device, result_path):
    train_losses, valid_losses = [], []
    min_val_loss = torch.inf

    weight_path = result_path + "weights/"

    for epoch in range(epochs):
        train_loss, valid_loss = 0., 0.

        model.train()
        #model.to(device)

        print('Train Batches')
        tr_bar = tqdm(train_batches, desc=f'epoch {epoch}')

        for images, labels in tr_bar:
            images = images.to(device)
            labels = labels.to(device)

            opt.zero_grad()

            outputs = model(images)

            loss = cri(outputs, labels)
            loss.backward()

            opt.step()

            train_loss += loss.item()*images.size(0)
            tr_bar.postfix = f"loss: {loss.item():.3f}"

            torch.cuda.empty_cache()
        
        train_loss /= len(train_batches.sampler)
        train_losses.append(train_loss)

        print("Validation Batches")
        val_bar = tqdm(valid_batches, desc=f'epoch {epoch}')

        model.eval()
        #model.to(device)

        for images, labels in val_bar:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = cri(outputs, labels)

            valid_loss += loss.item()*images.size(0)
            val_bar.postfix = f"loss: {loss.item():.3f}"
        
        valid_loss /= len(valid_batches.sampler)        
        valid_losses.append(valid_loss)

        print(f'Epoch #{epoch} training loss: {train_loss:.5f}\tvalidation loss: {valid_loss:.5f}\n')

        if valid_loss < min_val_loss:
            checkDir(weight_path)
            save_path = weight_path + f'epoch_{epoch}.pth'
            torch.save(model.state_dict(), save_path)
            print(f"Validation loss decreased ({min_val_loss:.5f} --> {valid_loss:.5f})\nSaving current weight at: {save_path})")
            min_val_loss = valid_loss
        elif epoch % 10 == 0:
            save_path = weight_path + f'epoch_{epoch}.pth'
            torch.save(model.state_dict(), save_path)
            print(f"Training epochs {epoch}/{epochs}\nSaving current weight at: {save_path}")
        elif epoch == epochs-1:
            save_path = weight_path + 'last.pth'
            torch.save(model.state_dict(), save_path)
    
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(valid_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(result_path+'loss_chart.png')

    print("Training End")

def run():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    num_classes = 7
    epochs = 10

    train_batches, valid_batches, weight = loadBatches(batch_size=32, is_train=True)
    weight = torch.tensor(weight, dtype=torch.float).to(device)
    
    model = loadModel(num_classes=num_classes, feature_extract=True)
    model = model.to(device)

    opt = optim.Adam(model.parameters(), lr=1e-4)
    cri = nn.CrossEntropyLoss(weight=weight)

    result_path = 'result/train/'
    result_path = checkDir(result_path)

    train(model, train_batches, valid_batches, epochs, opt, cri, device, result_path)

if __name__ == '__main__':
    run()
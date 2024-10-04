import os
import tqdm
import torch
import torch.nn as nn
import torch.optim as optim

from src.dataloader import loadBatches
from src.model import loadModel


def train(model, train_batches, valid_batches, epochs, opt, cri, device):
    train_losses, valid_losses = [], []
    min_val_loss = torch.inf

    for epoch in range(epochs):
        train_loss, valid_loss = 0., 0.

        model.train()

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
            tr_bar.postfix = f"{loss.item():.3f}"

            torch.cuda.empty_cache()
        
        train_loss /= len(train_batches.sampler)
        train_losses.append(train_loss)

        print("Validation Batches")
        val_bar = tqdm(valid_batches, desc=f'epoch {epoch}')

        model.eval()

        for images, labels in val_bar:
            outputs = model(images)

            loss = cri(outputs, labels)

            valid_loss += loss.item()*images.size(0)
        
        valid_loss /= len(valid_batches.sampler)        
        valid_losses.append(valid_loss)

        print(f'Epoch #{epoch} training loss: {train_loss}\tvalidation loss: {valid_loss}\n')

        # TODO: 모델 저장 로직, loss chart 시각화 추가
        if valid_loss < min_val_loss:


def run():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    num_classes = 7
    epochs = 100

    train_batches, valid_batches, weight = loadBatches(batch_size=32, is_train=True)
    weight = torch.tensor(weight, dtype=torch.float)
    
    model = loadModel(num_classes=num_classes, pretrained=True, feature_extract=True)
    model = model.to(device)

    opt = optim.Adam(model.parameters(), lr=1e-4)
    cri = nn.CrossEntropyLoss(weight=weight)

    train(model, train_batches, valid_batches, epochs, opt, cri, device)

if __name__ == '__main__':
    run()
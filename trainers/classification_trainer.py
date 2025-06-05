import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from torchmetrics.classification import MulticlassAccuracy, MulticlassAUROC, MulticlassF1Score
from torchmetrics import ConfusionMatrix

from utils.utils import save_state, load_state
from datasets import ClassificationDataset

class ClassifcationTrainer:
    def __init__(self, model: nn.Module, optim: torch.optim, loss_fn: nn.Module, num_classes:int, device:str='cpu'):
        self.model = model
        self.optim = optim
        self.loss_fn = loss_fn
        self.device = device
        self.num_classes = num_classes

        self.conf = ConfusionMatrix(task='multiclass',num_classes=num_classes).to(device)
        self.metrics = [
            MulticlassAccuracy(num_classes=num_classes, average='micro'),
            MulticlassF1Score(num_classes=num_classes, average='macro'),
            MulticlassAUROC(num_classes=num_classes),
        ]
        for metric in self.metrics:
            metric.to(device)


    def fit(self, num_epochs, train_db: ClassificationDataset, val_db: ClassificationDataset, batch_size, early_stop_patience=30, checkpoint_path=f'checkpoints/model.pt'):
        val_loader = DataLoader(val_db, batch_size=batch_size, shuffle=False, num_workers=2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim, T_max=num_epochs)
        train_losses = []
        val_losses = [np.inf]

        es_counter = 0
        
        for epoch in range(num_epochs):
            train_loader = DataLoader(train_db, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count())
            print(f'Epoch {epoch+1}/{num_epochs}')
            train_loss = self.train_epoch(train_loader)
            val_loss = self.evaluate(val_loader, use_progress_bar=False)
            scheduler.step()
            print(f'Learning Rate: {scheduler.get_last_lr()[-1]:0.4e}')
            
            es_counter += 1
            if val_loss < min(val_losses):
                print(f'val_loss improved from {np.min(val_losses):0.4f} to {val_loss:0.4f}')
                save_state(checkpoint_path, self.model, self.optim)
                es_counter = 0
            if es_counter >= early_stop_patience:
                print('Early Stopping')
                break

            train_losses.append(train_loss)
            val_losses.append(val_loss)
        self.model, self.optim = load_state(checkpoint_path, self.model, self.optim)



    def train_epoch(self, loader: DataLoader):
        self.model.train()
        losses = []

        for images, labels in tqdm(loader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            self.optim.zero_grad()
            with torch.autocast(self.device):
                outputs = self.model(images)
                loss = self.loss_fn(outputs, labels)
            losses.append(loss.item())
            loss.backward()
            self.optim.step()

        print(f'Train Loss: {np.mean(losses):0.4f}')
        return np.mean(losses)



    def evaluate(self, loader: DataLoader, conf=False, use_progress_bar=True):
        self.model.eval()
        losses = []

        self.conf.reset()
        for metric in self.metrics:
            metric.reset() 

        data_iter = loader
        if use_progress_bar:
            data_iter = tqdm(loader, leave=False)

        with torch.inference_mode():
            for images, labels in data_iter:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)
                loss = self.loss_fn(outputs, labels)
                losses.append(loss.item())

                for metric in self.metrics:
                    metric.update(outputs, labels)
                self.conf.update(outputs, labels)
        print(f'Eval Loss: {np.mean(losses):0.4f}', end=' | ')
        for metric in self.metrics:
            print(f'{metric.__class__.__name__[10:]}: {metric.compute().item():0.4f}', end=' | ')
        print()
        if conf:
            print(self.conf.compute().cpu().numpy() / 1206)
        return np.mean(losses)




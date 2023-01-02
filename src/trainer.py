import torch
import torch.nn as nn
import torch.optim as optim

class Trainer:
    def __init__(self, model, train_loader, val_loader, device="cpu"):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        self.optimizer = optim.Adam(model.parameters(), lr=1e-4)

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        for batch in self.train_loader:
            x, y = batch
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(x)
            loss = self.criterion(output.transpose(1, 2), y)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(self.train_loader)

    def evaluate(self):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in self.val_loader:
                x, y = batch
                x, y = x.to(self.device), y.to(self.device)
                output = self.model(x)
                loss = self.criterion(output.transpose(1, 2), y)
                total_loss += loss.item()
        return total_loss / len(self.val_loader)

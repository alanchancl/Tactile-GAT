import time
import torch
import numpy as np
from utils.device import get_device
from test import test

def loss_func(y_pred, y_true):
    return torch.nn.CrossEntropyLoss()(y_pred, y_true.long())

def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss, total_correct = 0, 0
    for x, labels, attack_labels in dataloader:
        x, attack_labels = [item.float().to(device) for item in [x, attack_labels]]
        optimizer.zero_grad()
        outputs = model(x).float()
        loss = loss_func(outputs, attack_labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_correct += (outputs.argmax(dim=1) == attack_labels).sum().item()
    return total_loss / len(dataloader.dataset), total_correct / len(dataloader.dataset)

def validate(model, dataloader, device):
    model.eval()
    with torch.no_grad():
        return test(model, dataloader)

def train(model, config, train_dataloader, val_dataloader):
    device = get_device()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['decay'])
    best_accuracy = 0
    patience, trials = 10, 0

    for epoch in range(config['epoch']):
        train_loss, train_acc = train_epoch(model, train_dataloader, optimizer, device)
        _, val_loss, val_acc, _, _, _ = validate(model, val_dataloader, device)
        
        print(f"Epoch {epoch+1}/{config['epoch']}, Train Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")

        if val_acc > best_accuracy:
            best_accuracy = val_acc
            torch.save(model.state_dict(), config['model_save_path'])
            trials = 0
        else:
            trials += 1
            if trials >= patience:
                print("Early stopping triggered.")
                break


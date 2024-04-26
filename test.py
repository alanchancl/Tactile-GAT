import torch
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from utils.device import get_device

def loss_func(y_pred, y_true):
    return torch.nn.CrossEntropyLoss()(y_pred, y_true.long())

def test(model, dataloader):
    device = get_device()
    model.to(device)
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    all_predicted, all_labels = [], []
    
    for x, _, labels in dataloader:
        x, labels = x.float().to(device), labels.long().to(device)
        
        with torch.no_grad():
            outputs = model(x)
            loss = loss_func(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()
        
        all_predicted.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        total_samples += labels.size(0)

    accuracy_rate = accuracy_score(all_predicted, all_labels)
    precision = precision_score(all_labels, all_predicted, average='macro')  # 'macro' averages over classes
    recall = recall_score(all_labels, all_predicted, average='macro')
    f1 = f1_score(all_labels, all_predicted, average='macro')
    avg_loss = total_loss / total_samples

    print(f'Test Loss: {avg_loss:.4f}, Accuracy: {accuracy_rate:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')

    return outputs, avg_loss, accuracy_rate, precision, recall, f1

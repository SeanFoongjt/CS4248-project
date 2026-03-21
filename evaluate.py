import torch
from sklearn.metrics import precision_score, recall_score, f1_score

def evaluate(model, loader, criterion, device):
    """Evaluates the model, ensuring all tensors are aligned to the correct hardware device."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device) # CRITICAL: Map test data to GPU/CPU
            out, _ = model(data.x, data.edge_index, data.edge_attr, data.edge_weight, data.batch) 
            probs = torch.sigmoid(out.squeeze(-1))
            loss = criterion(probs, data.y)
            total_loss += loss.item()
            
            predictions = (probs >= 0.5).float()
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(data.y.cpu().numpy())
            
    accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels) if all_labels else 0
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    return total_loss / len(loader), accuracy, precision, recall, f1
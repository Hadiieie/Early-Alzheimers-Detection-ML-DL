import torch
from sklearn.metrics import balanced_accuracy_score

def run_epoch(model, loader, loss_fn, device, optimizer=None):
    training = optimizer is not None
    model.train() if training else model.eval()
    total, y_true, y_pred = 0, [], []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if training: optimizer.zero_grad()
        with torch.set_grad_enabled(training):
            out = model(x)
            loss = loss_fn(out, y)
            if training:
                loss.backward()
                optimizer.step()
        total += loss.item() * len(y)
        y_true += y.cpu().tolist()
        y_pred += out.argmax(1).cpu().tolist()

    return total / len(loader.dataset), balanced_accuracy_score(y_true, y_pred)
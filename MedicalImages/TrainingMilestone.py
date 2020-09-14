from torch.utils.data import DataLoader

device = torch.device('cuda')
model.to(device)
optimizer = # choose optimizer here (e.g., Adam)
criterion = # choose loss function here (e.g., MSE)

for t in range(1, n_epochs + 1):
    model.train()
    for i, (src, tgt) in enumerate(train_loader):
        src, tgt = src.to(device), tgt.to(device)
        # fill in steps to t

    
    model.eval()
    with torch.no_grad():
        for src, tgt in valid_loader:
            src, tgt = src.to(device), tgt.to(device)
            # fill this in; note you don’t have to update parameters.

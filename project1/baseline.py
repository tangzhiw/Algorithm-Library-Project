import torch
from torch import nn, optim


def pytorch_logistic_regression(X, y, lr=0.01, batch_size=32, max_epochs=10):
    dataset = torch.utils.data.TensorDataset(torch.tensor(X, dtype=torch.float32),
                                             torch.tensor(y, dtype=torch.float32))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = nn.Linear(X.shape[1], 1, bias=False)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    losses = []
    for epoch in range(max_epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            logits = model(xb).squeeze()
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            logits = model(torch.tensor(X, dtype=torch.float32)).squeeze()
            loss_val = criterion(logits, torch.tensor(y, dtype=torch.float32))
            losses.append(loss_val.item())
    return losses

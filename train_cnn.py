import os, time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm

from utils import get_transforms, device
from model_cnn import ConvolutionalNetwork

def main(root="../Data/BEAD", batch_size=10, epochs=4, lr=1e-3, out="CNNModel.pt"):
    train_t, test_t = get_transforms()
    train_data = datasets.ImageFolder(os.path.join(root, "train"), transform=train_t)
    test_data  = datasets.ImageFolder(os.path.join(root, "test"), transform=test_t)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_data,  batch_size=batch_size, shuffle=False)

    d = device()
    model = ConvolutionalNetwork().to(d)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"Classes: {train_data.classes}")
    print(f"Training images: {len(train_data)} | Testing images: {len(test_data)}")
    start = time.time()

    for epoch in range(epochs):
        model.train()
        trn_corr = 0
        for b, (X, y) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")):
            X, y = X.to(d), y.to(d)
            y_pred = model(X)
            loss = criterion(y_pred, y)

            pred = torch.argmax(y_pred.data, dim=1)
            trn_corr += (pred == y).sum().item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (b+1) % 100 == 0:
                seen = (b+1) * batch_size
                acc = 100.0 * trn_corr / seen
                print(f"  batch {b+1:4d} [{seen:6d}/{len(train_data)}]  loss={loss.item():.6f}  acc={acc:6.2f}%")

        # validation
        model.eval()
        tst_corr = 0
        val_loss = 0.0
        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(d), y.to(d)
                y_val = model(X)
                pred = torch.argmax(y_val.data, dim=1)
                tst_corr += (pred == y).sum().item()
                val_loss += criterion(y_val, y).item()

        val_acc = 100.0 * tst_corr / len(test_data)
        print(f"Epoch {epoch+1}: val_loss={val_loss/len(test_loader):.4f}  val_acc={val_acc:.3f}%")

    torch.save(model.state_dict(), out)
    print(f"Saved model to {out}")
    print(f"Duration: {time.time()-start:.0f} seconds")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="../Data/BEAD")
    ap.add_argument("--batch_size", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="CNNModel.pt")
    args = ap.parse_args()
    main(**vars(args))

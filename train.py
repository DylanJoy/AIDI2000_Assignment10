# train.py
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import argparse
import os

def get_dataloaders(data_dir, batch_size=32, img_size=224):
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    val_tf = transforms.Compose([
        transforms.Resize(int(img_size*1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    train_ds = datasets.ImageFolder(os.path.join(data_dir,'train'), transform=train_tf)
    val_ds = datasets.ImageFolder(os.path.join(data_dir,'val'), transform=val_tf)
    return DataLoader(train_ds, batch_size=batch_size, shuffle=True), DataLoader(val_ds, batch_size=batch_size), train_ds.classes

def build_model(num_classes):
    model = models.resnet50(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    # replace head
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, num_classes)
    )
    return model

def train(args):
    train_loader, val_loader, classes = get_dataloaders(args.data_dir, args.batch_size, args.img_size)
    model = build_model(len(classes)).to(args.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-4)

    best_val = 0.0
    for epoch in range(args.epochs):
        model.train()
        total, correct = 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(args.device), labels.to(args.device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            preds = out.argmax(1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
        train_acc = correct/total

        # val
        model.eval()
        total, correct = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(args.device), labels.to(args.device)
                out = model(imgs)
                preds = out.argmax(1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()
        val_acc = correct/total
        print(f"Epoch {epoch}: train_acc={train_acc:.3f}, val_acc={val_acc:.3f}")
        if val_acc > best_val:
            best_val = val_acc
            torch.save({'model_state': model.state_dict(), 'classes': classes}, args.save_path)
    print("Done. Best val:", best_val)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--save_path", default="model.pt")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    train(args)

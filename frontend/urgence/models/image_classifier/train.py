import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

BASE     = r"C:\Users\ismai\OneDrive\Desktop\urgence_system"
DATA_DIR = os.path.join(BASE, "data", "final")
SAVE_DIR = os.path.join(BASE, "models", "image_classifier")
os.makedirs(SAVE_DIR, exist_ok=True)

BATCH_SIZE = 32
EPOCHS     = 25
LR         = 1e-4
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

val_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for imgs, labels in tqdm(loader, leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            if train:
                optimizer.zero_grad()
            out  = model(imgs)
            loss = criterion(out, labels)
            if train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * imgs.size(0)
            correct    += (out.argmax(1) == labels).sum().item()
            total      += imgs.size(0)
    return total_loss / total, correct / total


if __name__ == '__main__':
    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    train_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "train", "images"), transform=train_tf)
    val_ds   = datasets.ImageFolder(os.path.join(DATA_DIR, "val",   "images"), transform=val_tf)
    test_ds  = datasets.ImageFolder(os.path.join(DATA_DIR, "test",  "images"), transform=val_tf)

    CLASSES     = train_ds.classes
    NUM_CLASSES = len(CLASSES)
    print(f"\nClasses: {CLASSES}")
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    # num_workers=0 — obligatoire f Windows
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    counts  = [0] * NUM_CLASSES
    for _, lbl in train_ds.samples:
        counts[lbl] += 1
    total   = sum(counts)
    weights = torch.tensor(
        [total / (NUM_CLASSES * c) for c in counts],
        dtype=torch.float
    ).to(DEVICE)

    print(f"\nDistribution train:")
    for i, cls in enumerate(CLASSES):
        print(f"  {cls}: {counts[i]} images | weight: {weights[i]:.2f}")

    model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
    in_f  = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_f, NUM_CLASSES)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3
    )

    history  = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_acc = 0.0

    print("\n━━━ Training MobileNetV3 ━━━\n")

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, DEVICE, train=True)
        vl_loss, vl_acc = run_epoch(model, val_loader,   criterion, optimizer, DEVICE, train=False)
        scheduler.step(vl_acc)

        history['train_loss'].append(tr_loss)
        history['val_loss'].append(vl_loss)
        history['train_acc'].append(tr_acc)
        history['val_acc'].append(vl_acc)

        flag = ""
        if vl_acc > best_acc:
            best_acc = vl_acc
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_model.pth"))
            flag = "  ✅ saved"

        print(f"Epoch {epoch:02d}/{EPOCHS} | "
              f"Train: loss={tr_loss:.4f} acc={tr_acc:.4f} | "
              f"Val: loss={vl_loss:.4f} acc={vl_acc:.4f}{flag}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("MobileNetV3 — Training Curves")
    axes[0].plot(history['train_loss'], label='Train')
    axes[0].plot(history['val_loss'],   label='Val')
    axes[0].set_title('Loss')
    axes[0].legend()
    axes[1].plot(history['train_acc'], label='Train')
    axes[1].plot(history['val_acc'],   label='Val')
    axes[1].set_title('Accuracy')
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "training_curves.png"), dpi=150)
    plt.close()

    config = {
        'classes':      CLASSES,
        'num_classes':  NUM_CLASSES,
        'img_size':     224,
        'best_val_acc': round(best_acc, 4),
        'note':         'P1 handled by safety_net only',
    }
    with open(os.path.join(SAVE_DIR, "config.json"), 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\n✅ Best Val Accuracy: {best_acc:.4f}")
    print(f"   Model  → {SAVE_DIR}/best_model.pth")
    print(f"   Curves → {SAVE_DIR}/training_curves.png")
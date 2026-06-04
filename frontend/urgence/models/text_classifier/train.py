import os
import json
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import CamembertTokenizer, CamembertForSequenceClassification
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

BASE     = r"C:\Users\ismai\OneDrive\Desktop\urgence_system"
SAVE_DIR = os.path.join(BASE, "models", "text_classifier")
os.makedirs(SAVE_DIR, exist_ok=True)

CLASSES   = ['P1', 'P2', 'P3', 'P4']
LABEL_MAP = {'P1': 0, 'P2': 1, 'P3': 2, 'P4': 3}
MAX_LEN   = 256
BATCH_SIZE = 16
EPOCHS    = 10
LR        = 2e-5
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TriageDataset(Dataset):
    def __init__(self, csv_path, tokenizer):
        df = pd.read_csv(csv_path)
        self.texts  = df['texte'].tolist()
        self.labels = df['urgence'].map(LABEL_MAP).tolist()
        self.tok    = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tok(
            self.texts[idx],
            max_length=MAX_LEN,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        return {
            'input_ids':      enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'label':          torch.tensor(self.labels[idx], dtype=torch.long),
        }


def run_epoch(model, loader, criterion, optimizer, scheduler, device, train=True):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in tqdm(loader, leave=False):
            ids    = batch['input_ids'].to(device)
            mask   = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            if train:
                optimizer.zero_grad()
            out  = model(input_ids=ids, attention_mask=mask)
            loss = criterion(out.logits, labels)
            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
            total_loss += loss.item() * ids.size(0)
            correct    += (out.logits.argmax(1) == labels).sum().item()
            total      += ids.size(0)
    return total_loss / total, correct / total


if __name__ == '__main__':
    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print("\nChargement CamemBERT...")
    tokenizer = CamembertTokenizer.from_pretrained("camembert-base")
    model     = CamembertForSequenceClassification.from_pretrained(
        "camembert-base", num_labels=4
    ).to(DEVICE)

    data_dir     = os.path.join(BASE, "data", "final")
    train_ds     = TriageDataset(os.path.join(data_dir, "train", "text.csv"), tokenizer)
    val_ds       = TriageDataset(os.path.join(data_dir, "val",   "text.csv"), tokenizer)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    df_train = pd.read_csv(os.path.join(data_dir, "train", "text.csv"))
    total_n  = len(df_train)
    counts   = df_train['urgence'].value_counts()
    weights  = torch.tensor(
        [total_n / (4 * counts.get(c, 1)) for c in CLASSES],
        dtype=torch.float
    ).to(DEVICE)
    print(f"Class weights: {weights.cpu().numpy().round(2)}")

    criterion   = nn.CrossEntropyLoss(weight=weights)
    optimizer   = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS
    scheduler   = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps,
    )

    history  = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_acc = 0.0

    print("\n━━━ Training CamemBERT ━━━\n")

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, scheduler, DEVICE, train=True)
        vl_loss, vl_acc = run_epoch(model, val_loader,   criterion, optimizer, scheduler, DEVICE, train=False)

        history['train_loss'].append(tr_loss)
        history['val_loss'].append(vl_loss)
        history['train_acc'].append(tr_acc)
        history['val_acc'].append(vl_acc)

        flag = ""
        if vl_acc > best_acc:
            best_acc = vl_acc
            model.save_pretrained(os.path.join(SAVE_DIR, "best_model"))
            tokenizer.save_pretrained(os.path.join(SAVE_DIR, "best_model"))
            flag = "  ✅ saved"

        print(f"Epoch {epoch:02d}/{EPOCHS} | "
              f"Train: loss={tr_loss:.4f} acc={tr_acc:.4f} | "
              f"Val: loss={vl_loss:.4f} acc={vl_acc:.4f}{flag}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("CamemBERT — Training Curves")
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
        'label_map':    LABEL_MAP,
        'max_len':      MAX_LEN,
        'best_val_acc': round(best_acc, 4),
        'model_name':   'camembert-base',
    }
    with open(os.path.join(SAVE_DIR, "config.json"), 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\n✅ Best Val Accuracy: {best_acc:.4f}")
    print(f"   Model  → {SAVE_DIR}/best_model/")
    print(f"   Curves → {SAVE_DIR}/training_curves.png")
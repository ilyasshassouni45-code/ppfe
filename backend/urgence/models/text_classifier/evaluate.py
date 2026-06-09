import os
import json
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import CamembertTokenizer, CamembertForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

BASE     = r"C:\Users\ismai\OneDrive\Desktop\urgence_system"
SAVE_DIR = os.path.join(BASE, "models", "text_classifier")
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TriageDataset(Dataset):
    def __init__(self, csv_path, tokenizer, label_map, max_len):
        df = pd.read_csv(csv_path)
        self.texts  = df['texte'].tolist()
        self.labels = df['urgence'].map(label_map).tolist()
        self.tok    = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tok(
            self.texts[idx],
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        return {
            'input_ids':      enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'label':          torch.tensor(self.labels[idx], dtype=torch.long),
        }


if __name__ == '__main__':
    with open(os.path.join(SAVE_DIR, "config.json")) as f:
        config = json.load(f)

    CLASSES   = config['classes']
    LABEL_MAP = config['label_map']
    MAX_LEN   = config['max_len']

    model_path  = os.path.join(SAVE_DIR, "best_model")
    tokenizer   = CamembertTokenizer.from_pretrained(model_path)
    model       = CamembertForSequenceClassification.from_pretrained(
        model_path, num_labels=4
    ).to(DEVICE).eval()

    test_ds     = TriageDataset(
        os.path.join(BASE, "data", "final", "test", "text.csv"),
        tokenizer, LABEL_MAP, MAX_LEN
    )
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            ids    = batch['input_ids'].to(DEVICE)
            mask   = batch['attention_mask'].to(DEVICE)
            labels = batch['label']
            preds  = model(input_ids=ids, attention_mask=mask).logits.argmax(1).cpu()
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())

    print("\n📊 Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=CLASSES))

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
                xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title("Confusion Matrix — Text Classifier")
    plt.ylabel("Réel")
    plt.xlabel("Prédit")
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "confusion_matrix.png"), dpi=150)
    plt.close()

    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    print(f"✅ Test Accuracy: {acc:.4f}")
    print(f"   Confusion matrix → {SAVE_DIR}/confusion_matrix.png")
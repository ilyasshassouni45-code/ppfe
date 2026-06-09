import sys
import os
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, r"C:\Users\ismai\OneDrive\Desktop\urgence_system")
from models.fusion.fusion import UrgenceClassifier

BASE     = r"C:\Users\ismai\OneDrive\Desktop\urgence_system"
SAVE_DIR = os.path.join(BASE, "models", "fusion")
os.makedirs(SAVE_DIR, exist_ok=True)
CLASSES  = ['P1', 'P2', 'P3', 'P4']

if __name__ == '__main__':
    clf = UrgenceClassifier()
    df  = pd.read_csv(os.path.join(BASE, "data", "final", "test", "text.csv"))

    all_preds, all_labels = [], []
    print(f"Évaluation sur {len(df)} exemples...\n")

    for i, row in df.iterrows():
        result = clf.predict(row['texte'])
        all_preds.append(result['niveau'])
        all_labels.append(row['urgence'])
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(df)}...")

    print("\n📊 Classification Report — Fusion:")
    print(classification_report(all_labels, all_preds, target_names=CLASSES))

    label_map = {c: i for i, c in enumerate(CLASSES)}
    y_true = [label_map[l] for l in all_labels]
    y_pred = [label_map[p] for p in all_preds]

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
                xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title("Confusion Matrix — Fusion System")
    plt.ylabel("Réel")
    plt.xlabel("Prédit")
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "confusion_matrix.png"), dpi=150)
    plt.close()

    acc = np.mean([p == l for p, l in zip(all_preds, all_labels)])
    print(f"✅ Fusion Accuracy: {acc:.4f}")
    print(f"   Confusion matrix → {SAVE_DIR}/confusion_matrix.png")
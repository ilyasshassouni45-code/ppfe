import os
import json
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

BASE     = r"C:\Users\ismai\OneDrive\Desktop\urgence_system"
SAVE_DIR = os.path.join(BASE, "models", "image_classifier")
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    with open(os.path.join(SAVE_DIR, "config.json")) as f:
        config = json.load(f)

    CLASSES     = config['classes']
    NUM_CLASSES = config['num_classes']

    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    test_ds     = datasets.ImageFolder(
        os.path.join(BASE, "data", "final", "test", "images"),
        transform=val_tf
    )
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)

    model = models.mobilenet_v3_large(weights=None)
    in_f  = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_f, NUM_CLASSES)
    model.load_state_dict(
        torch.load(os.path.join(SAVE_DIR, "best_model.pth"), map_location=DEVICE)
    )
    model = model.to(DEVICE).eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            preds = model(imgs.to(DEVICE)).argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    print("\n📊 Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=CLASSES))

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title("Confusion Matrix — Image Classifier")
    plt.ylabel("Réel")
    plt.xlabel("Prédit")
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "confusion_matrix.png"), dpi=150)
    plt.close()

    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    print(f"✅ Test Accuracy: {acc:.4f}")
    print(f"   Confusion matrix → {SAVE_DIR}/confusion_matrix.png")
import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from transformers import CamembertTokenizer, CamembertForSequenceClassification
from PIL import Image

sys.path.insert(0, r"C:\Users\ismai\OneDrive\Desktop\urgence_system")
from safety_net import check as safety_check

BASE      = r"C:\Users\ismai\OneDrive\Desktop\urgence_system"
IMG_DIR   = os.path.join(BASE, "models", "image_classifier")
TEXT_DIR  = os.path.join(BASE, "models", "text_classifier")
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ALL_CLASSES = ['P1', 'P2', 'P3', 'P4']

ACTIONS = {
    'P1': 'Admission immédiate — urgence absolue',
    'P2': 'Prise en charge < 10 minutes',
    'P3': 'Prise en charge < 60 minutes',
    'P4': "Consultation standard — salle d'attente",
}
COLORS = {'P1': 'ROUGE', 'P2': 'ORANGE', 'P3': 'JAUNE', 'P4': 'VERT'}

IMG_WEIGHT  = 0.40
TEXT_WEIGHT = 0.60

img_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class UrgenceClassifier:

    def __init__(self):
        print("Chargement modèles...", flush=True)

        # ── Image model ───────────────────────────────────────────────────────
        with open(os.path.join(IMG_DIR, "config.json")) as f:
            img_config = json.load(f)

        self.img_classes  = img_config['classes']   # ['P2','P3','P4']
        self.img_n        = img_config['num_classes']

        self.img_model = models.mobilenet_v3_large(weights=None)
        in_f = self.img_model.classifier[3].in_features
        self.img_model.classifier[3] = nn.Linear(in_f, self.img_n)
        self.img_model.load_state_dict(
            torch.load(os.path.join(IMG_DIR, "best_model.pth"), map_location=DEVICE)
        )
        self.img_model = self.img_model.to(DEVICE).eval()

        # ── Text model ────────────────────────────────────────────────────────
        with open(os.path.join(TEXT_DIR, "config.json")) as f:
            txt_config = json.load(f)

        self.txt_label_map = txt_config['label_map']  # {'P1':0,'P2':1,'P3':2,'P4':3}
        self.txt_max_len   = txt_config['max_len']

        model_path       = os.path.join(TEXT_DIR, "best_model")
        self.tokenizer   = CamembertTokenizer.from_pretrained(model_path)
        self.text_model  = CamembertForSequenceClassification.from_pretrained(
            model_path, num_labels=4
        ).to(DEVICE).eval()

        print("✅ Modèles chargés\n", flush=True)

    # ── Prédiction texte ──────────────────────────────────────────────────────

    def _predict_text(self, text: str) -> torch.Tensor:
        enc = self.tokenizer(
            text,
            max_length=self.txt_max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        with torch.no_grad():
            logits = self.text_model(
                input_ids=enc['input_ids'].to(DEVICE),
                attention_mask=enc['attention_mask'].to(DEVICE)
            ).logits
        return F.softmax(logits, dim=1).squeeze(0).cpu()  # [P1,P2,P3,P4]

    # ── Prédiction image ──────────────────────────────────────────────────────

    def _predict_image(self, image_path: str) -> torch.Tensor | None:
        if not image_path or not os.path.exists(image_path):
            return None
        try:
            img    = Image.open(image_path).convert('RGB')
            tensor = img_transform(img).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                logits = self.img_model(tensor)
            img_scores = F.softmax(logits, dim=1).squeeze(0).cpu()  # [P2,P3,P4]

            # Mapper vers 4 classes [P1,P2,P3,P4] — P1=0 car pas d'images P1
            mapped = torch.zeros(4)
            for i, cls in enumerate(self.img_classes):
                idx = ALL_CLASSES.index(cls)
                mapped[idx] = img_scores[i]

            return mapped
        except Exception as e:
            print(f"  Image ignorée: {e}", flush=True)
            return None

    # ── Prédiction finale ─────────────────────────────────────────────────────

    def predict(self, symptomes: str, image_path: str = None) -> dict:
        # Couche 1 — Safety Net
        sn = safety_check(symptomes)
        if sn['is_p1']:
            return {
                'niveau':    'P1',
                'confiance': 100,
                'action':    ACTIONS['P1'],
                'couleur':   COLORS['P1'],
                'source':    'safety_net',
                'keyword':   sn['keyword'],
                'scores':    {'P1': 1.0, 'P2': 0.0, 'P3': 0.0, 'P4': 0.0},
            }

        # Couche 2 — AI
        text_scores = self._predict_text(symptomes)    # tensor [4]
        img_scores  = self._predict_image(image_path)  # tensor [4] ou None

        if img_scores is not None:
            final = TEXT_WEIGHT * text_scores + IMG_WEIGHT * img_scores
            source = 'image_text'
        else:
            final  = text_scores
            source = 'text_only'

        # Normaliser
        final = final / final.sum()

        idx       = final.argmax().item()
        niveau    = ALL_CLASSES[idx]
        confiance = int(final[idx].item() * 100)

        return {
            'niveau':    niveau,
            'confiance': confiance,
            'action':    ACTIONS[niveau],
            'couleur':   COLORS[niveau],
            'source':    source,
            'keyword':   None,
            'scores': {ALL_CLASSES[i]: round(final[i].item(), 4) for i in range(4)},
        }
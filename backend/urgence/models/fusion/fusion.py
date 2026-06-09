import json
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms
from transformers import CamembertForSequenceClassification, CamembertTokenizer


# ============================================================
# Paths
# ============================================================

# fusion.py is here:
# backend/urgence/models/fusion/fusion.py
#
# parents[0] = fusion
# parents[1] = models
# parents[2] = urgence
URGENCE_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = URGENCE_DIR / "models"

IMG_DIR = MODELS_DIR / "image_classifier"
TEXT_DIR = MODELS_DIR / "text_classifier"

# Allow importing safety_net.py from backend/urgence
sys.path.insert(0, str(URGENCE_DIR))

from safety_net import check as safety_check  # noqa: E402


# ============================================================
# Device
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Classes and labels
# ============================================================

ALL_CLASSES = ["P1", "P2", "P3", "P4"]

ACTIONS = {
    "P1": "Admission immédiate — urgence absolue",
    "P2": "Prise en charge < 10 minutes",
    "P3": "Prise en charge < 60 minutes",
    "P4": "Consultation standard — salle d'attente",
}

COLORS = {
    "P1": "ROUGE",
    "P2": "ORANGE",
    "P3": "JAUNE",
    "P4": "VERT",
}


# ============================================================
# Fusion weights
# ============================================================

IMG_WEIGHT = 0.40
TEXT_WEIGHT = 0.60


# ============================================================
# Image preprocessing
# ============================================================

img_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


class UrgenceClassifier:
    def __init__(self):
        print("Chargement des modèles...", flush=True)

        self.image_available = False
        self.text_available = False

        self.img_classes = []
        self.img_n = 0
        self.img_model = None

        self.txt_label_map = {}
        self.txt_max_len = 256
        self.tokenizer = None
        self.text_model = None

        self._load_image_model()
        self._load_text_model()

        if not self.image_available and not self.text_available:
            raise RuntimeError(
                "Aucun modèle AI n'a été chargé. "
                "Vérifie best_model.pth pour l'image et best_model/ pour le texte."
            )

        print("Modèles chargés avec succès.", flush=True)

    # ========================================================
    # Load image model
    # ========================================================

    def _load_image_model(self):
        config_path = IMG_DIR / "config.json"
        model_path = IMG_DIR / "best_model.pth"

        if not config_path.exists():
            print(f"Image config introuvable: {config_path}", flush=True)
            return

        if not model_path.exists():
            print(f"Image model introuvable: {model_path}", flush=True)
            return

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                img_config = json.load(f)

            self.img_classes = img_config.get("classes", ["P2", "P3", "P4"])
            self.img_n = int(img_config.get("num_classes", len(self.img_classes)))

            self.img_model = models.mobilenet_v3_large(weights=None)

            in_features = self.img_model.classifier[3].in_features
            self.img_model.classifier[3] = nn.Linear(in_features, self.img_n)

            checkpoint = torch.load(model_path, map_location=DEVICE)

            # Normal case: checkpoint is directly the state_dict
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                checkpoint = checkpoint["state_dict"]

            # Remove "module." prefix if model was trained with DataParallel
            if isinstance(checkpoint, dict):
                cleaned_checkpoint = {}
                for key, value in checkpoint.items():
                    new_key = key.replace("module.", "")
                    cleaned_checkpoint[new_key] = value
                checkpoint = cleaned_checkpoint

            self.img_model.load_state_dict(checkpoint)
            self.img_model = self.img_model.to(DEVICE)
            self.img_model.eval()

            self.image_available = True
            print("Image model chargé.", flush=True)

        except Exception as e:
            self.image_available = False
            self.img_model = None
            print(f"Erreur chargement image model: {e}", flush=True)

    # ========================================================
    # Load text model
    # ========================================================

    def _load_text_model(self):
        config_path = TEXT_DIR / "config.json"
        model_path = TEXT_DIR / "best_model"

        if not config_path.exists():
            print(f"Text config introuvable: {config_path}", flush=True)
            return

        if not model_path.exists():
            print(f"Text model folder introuvable: {model_path}", flush=True)
            return

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                txt_config = json.load(f)

            self.txt_label_map = txt_config.get(
                "label_map",
                {"P1": 0, "P2": 1, "P3": 2, "P4": 3},
            )
            self.txt_max_len = int(txt_config.get("max_len", 256))

            self.tokenizer = CamembertTokenizer.from_pretrained(str(model_path))

            self.text_model = CamembertForSequenceClassification.from_pretrained(
                str(model_path),
                num_labels=4,
            )

            self.text_model = self.text_model.to(DEVICE)
            self.text_model.eval()

            self.text_available = True
            print("Text model chargé.", flush=True)

        except Exception as e:
            self.text_available = False
            self.tokenizer = None
            self.text_model = None
            print(f"Erreur chargement text model: {e}", flush=True)

    # ========================================================
    # Text prediction
    # ========================================================

    def _predict_text(self, text: str) -> Optional[torch.Tensor]:
        if not self.text_available:
            return None

        if not text:
            return None

        try:
            enc = self.tokenizer(
                text,
                max_length=self.txt_max_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            input_ids = enc["input_ids"].to(DEVICE)
            attention_mask = enc["attention_mask"].to(DEVICE)

            with torch.no_grad():
                logits = self.text_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                ).logits

            scores = F.softmax(logits, dim=1).squeeze(0).cpu()

            # Expected output order: [P1, P2, P3, P4]
            return scores

        except Exception as e:
            print(f"Erreur prédiction texte: {e}", flush=True)
            return None

    # ========================================================
    # Image prediction
    # ========================================================

    def _predict_image(self, image_path: Optional[str]) -> Optional[torch.Tensor]:
        if not self.image_available:
            return None

        if not image_path:
            return None

        image_path = Path(image_path)

        if not image_path.exists():
            return None

        try:
            img = Image.open(image_path).convert("RGB")
            tensor = img_transform(img).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                logits = self.img_model(tensor)

            img_scores = F.softmax(logits, dim=1).squeeze(0).cpu()

            # Image model has only P2, P3, P4.
            # We map it to [P1, P2, P3, P4].
            mapped = torch.zeros(4)

            for i, cls in enumerate(self.img_classes):
                if cls in ALL_CLASSES:
                    idx = ALL_CLASSES.index(cls)
                    mapped[idx] = img_scores[i]

            return mapped

        except Exception as e:
            print(f"Image ignorée: {e}", flush=True)
            return None

    # ========================================================
    # Final prediction
    # ========================================================

    def predict(self, symptomes: str, image_path: Optional[str] = None) -> dict:
        symptomes = symptomes or ""

        # ----------------------------------------------------
        # Layer 1: Safety net
        # This always has priority.
        # ----------------------------------------------------
        sn = safety_check(symptomes)

        if sn.get("is_p1"):
            return {
                "niveau": "P1",
                "confiance": 100,
                "action": ACTIONS["P1"],
                "couleur": COLORS["P1"],
                "source": "safety_net",
                "keyword": sn.get("keyword"),
                "scores": {
                    "P1": 1.0,
                    "P2": 0.0,
                    "P3": 0.0,
                    "P4": 0.0,
                },
            }

        # ----------------------------------------------------
        # Layer 2: AI models
        # ----------------------------------------------------
        text_scores = self._predict_text(symptomes)
        img_scores = self._predict_image(image_path)

        if text_scores is not None and img_scores is not None:
            final = TEXT_WEIGHT * text_scores + IMG_WEIGHT * img_scores
            source = "image_text"

        elif text_scores is not None:
            final = text_scores
            source = "text_only"

        elif img_scores is not None:
            final = img_scores
            source = "image_only"

        else:
            # No AI available for this request.
            # Do not return random values.
            return {
                "niveau": "P4",
                "confiance": 0,
                "action": ACTIONS["P4"],
                "couleur": COLORS["P4"],
                "source": "no_model_available",
                "keyword": None,
                "scores": {
                    "P1": 0.0,
                    "P2": 0.0,
                    "P3": 0.0,
                    "P4": 1.0,
                },
            }

        # Avoid division by zero
        total = final.sum().item()
        if total > 0:
            final = final / final.sum()

        idx = final.argmax().item()
        niveau = ALL_CLASSES[idx]
        confiance = int(final[idx].item() * 100)

        return {
            "niveau": niveau,
            "confiance": confiance,
            "action": ACTIONS[niveau],
            "couleur": COLORS[niveau],
            "source": source,
            "keyword": None,
            "scores": {
                ALL_CLASSES[i]: round(final[i].item(), 4)
                for i in range(len(ALL_CLASSES))
            },
        }


# ============================================================
# Local test
# ============================================================

if __name__ == "__main__":
    classifier = UrgenceClassifier()

    result = classifier.predict(
        "Patient avec rougeur, douleur, gonflement et démangeaisons depuis 2 jours."
    )

    print(json.dumps(result, indent=2, ensure_ascii=False))
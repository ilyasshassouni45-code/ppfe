from fastapi import APIRouter, UploadFile, File, HTTPException
from ultralytics import YOLO
import numpy as np
import tempfile
import os
import sys

router = APIRouter(prefix="/ai", tags=["AI Tools"])

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "..", "outil.ia")

sys.path.insert(0, MODELS_DIR)
from gags import calculer_gags


def load_model(model_name):
    path = os.path.join(MODELS_DIR, model_name)
    if not os.path.exists(path):
        raise HTTPException(status_code=500, detail=f"Modele {model_name} introuvable")
    return YOLO(path)


async def save_upload(file: UploadFile):
    suffix = os.path.splitext(file.filename)[1]
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(await file.read())
    tmp.close()
    return tmp.name


# ── 1. Melanome ──────────────────────────────────────────────────────────────

@router.post("/melanome")
async def analyser_melanome(file: UploadFile = File(...)):
    tmp_path = await save_upload(file)
    try:
        model = load_model("melanoma_classifier.pt")
        results = model(tmp_path)[0]
        classe = results.names[results.probs.top1]
        confiance = float(results.probs.top1conf) * 100
        risk_levels = {
            'Benign':    'Faible risque - Surveillance',
            'Malignant': 'RISQUE ELEVE - Biopsie urgente'
        }
        return {
            "outil": "melanome",
            "classe": classe,
            "confiance": round(confiance, 1),
            "niveau_risque": "ELEVE" if classe == "Malignant" else "FAIBLE",
            "recommandation": risk_levels.get(classe, "Inconnu")
        }
    finally:
        os.unlink(tmp_path)


# ── 2. Acne ──────────────────────────────────────────────────────────────────

@router.post("/acne")
async def analyser_acne(file: UploadFile = File(...)):
    tmp_path = await save_upload(file)
    try:
        model = load_model("acne_classifier.pt")
        results = model(tmp_path)[0]
        classe = results.names[results.probs.top1]
        confiance = float(results.probs.top1conf) * 100
        severity_levels = {
            'Clear':    'Peau saine - Aucun traitement',
            'Mild':     'Acne legere - Traitement topique',
            'Moderate': 'Acne moderee - Consultation recommandee',
            'Severe':   'Acne severe - Traitement urgent'
        }
        return {
            "outil": "acne",
            "classe": classe,
            "confiance": round(confiance, 1),
            "niveau_risque": classe,
            "recommandation": severity_levels.get(classe, "Inconnu")
        }
    finally:
        os.unlink(tmp_path)


# ── 3. GAGS ───────────────────────────────────────────────────────────────────

@router.post("/gags")
async def calculer_gags_score(data: dict):
    try:
        forehead    = int(data.get("forehead", 0))
        right_cheek = int(data.get("right_cheek", 0))
        left_cheek  = int(data.get("left_cheek", 0))
        nose        = int(data.get("nose", 0))
        chin        = int(data.get("chin", 0))

        for val in [forehead, right_cheek, left_cheek, nose, chin]:
            if not (0 <= val <= 4):
                raise HTTPException(status_code=400, detail="Grades doivent etre entre 0 et 4")

        resultat = calculer_gags(forehead, right_cheek, left_cheek, nose, chin)
        return {
            "outil": "gags",
            "score": resultat["score"],
            "classe": resultat["severite"],
            "confiance": 100.0,
            "niveau_risque": resultat["severite"],
            "recommandation": resultat["traitement"]
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ── 4. Skin Lesion ────────────────────────────────────────────────────────────

@router.post("/skin-lesion")
async def analyser_skin_lesion(file: UploadFile = File(...)):
    tmp_path = await save_upload(file)
    try:
        model = load_model("skin_lesion_classifier.pt")
        results = model(tmp_path)[0]
        probs = results.probs.data.cpu().numpy()
        names = results.names

        descriptions = {
            'Dermatitis': 'Inflammation cutanee - Traitement topique',
            'Eczema':     'Eczema - Creme hydratante + corticoides',
            'Rosacea':    'Rosacea - Traitement medical recommande',
            'Normal':     'Peau saine - Aucun traitement'
        }

        top3_idx = [i for i in np.argsort(probs)[::-1]
                    if names[i] != 'Psoriasis'][:3]

        top3 = [
            {
                "classe": names[i],
                "confiance": round(float(probs[i]) * 100, 1),
                "recommandation": descriptions.get(names[i], "")
            }
            for i in top3_idx
        ]

        return {
            "outil": "skin_lesion",
            "classe": top3[0]["classe"],
            "confiance": top3[0]["confiance"],
            "niveau_risque": top3[0]["classe"],
            "recommandation": top3[0]["recommandation"],
            "top3": top3
        }
    finally:
        os.unlink(tmp_path)
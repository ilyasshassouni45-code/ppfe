import os
import sys
import base64
import tempfile
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session
from typing import Optional

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from api.database import get_db
from api.models_db import Triage
from api.websocket_manager import manager
from queue_manager import add_patient

router = APIRouter(prefix="/triage", tags=["Triage"])


class TriageRequest(BaseModel):
    nom:          str
    age:          Optional[int] = None
    sexe:         Optional[str] = None
    symptomes:    str
    antecedents:  Optional[str] = None
    image_base64: Optional[str] = None


class AdmissionDirecteRequest(BaseModel):
    nom:  str
    age:  Optional[int] = None
    sexe: Optional[str] = None
    note: Optional[str] = None


def get_classifier():
    from models.fusion.fusion import UrgenceClassifier
    if not hasattr(get_classifier, "_instance"):
        get_classifier._instance = UrgenceClassifier()
    return get_classifier._instance


@router.post("/classifier")
async def classifier_patient(req: TriageRequest, db: Session = Depends(get_db)):
    image_path = None

    if req.image_base64:
        try:
            data = base64.b64decode(req.image_base64)
            tmp  = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
            tmp.write(data)
            tmp.close()
            image_path = tmp.name
        except Exception:
            image_path = None

    result = get_classifier().predict(req.symptomes, image_path)

    if image_path and os.path.exists(image_path):
        os.unlink(image_path)

    triage = Triage(
        patient_nom       = req.nom,
        patient_age       = req.age,
        patient_sexe      = req.sexe,
        symptomes         = req.symptomes,
        niveau_urgence    = result['niveau'],
        confiance         = result['confiance'],
        action            = result['action'],
        couleur           = result['couleur'],
        source            = result['source'],
        keyword_p1        = result.get('keyword'),
        has_image         = image_path is not None,
        score_p1          = result['scores'].get('P1', 0.0),
        score_p2          = result['scores'].get('P2', 0.0),
        score_p3          = result['scores'].get('P3', 0.0),
        score_p4          = result['scores'].get('P4', 0.0),
    )
    db.add(triage)
    db.commit()
    db.refresh(triage)

    patient_data = {
        'nom':       req.nom,
        'age':       req.age,
        'sexe':      req.sexe,
        'niveau':    result['niveau'],
        'couleur':   result['couleur'],
        'action':    result['action'],
        'status':    'en_attente',
        'triage_id': triage.id,
    }
    add_patient(triage.id, result['niveau'], patient_data)

    await manager.broadcast({'event': 'nouveau_patient', 'patient': patient_data})

    return {
        'triage_id': triage.id,
        'niveau':    result['niveau'],
        'confiance': result['confiance'],
        'action':    result['action'],
        'couleur':   result['couleur'],
        'source':    result['source'],
        'keyword':   result.get('keyword'),
        'scores':    result['scores'],
    }


@router.post("/admission-directe")
async def admission_directe(req: AdmissionDirecteRequest, db: Session = Depends(get_db)):
    triage = Triage(
        patient_nom       = req.nom,
        patient_age       = req.age,
        patient_sexe      = req.sexe,
        symptomes         = req.note or "Admission directe — état critique visible",
        niveau_urgence    = 'P1',
        confiance         = 100,
        action            = "Admission immédiate — urgence absolue",
        couleur           = 'ROUGE',
        source            = 'admission_directe',
        admission_directe = True,
    )
    db.add(triage)
    db.commit()
    db.refresh(triage)

    patient_data = {
        'nom':              req.nom,
        'age':              req.age,
        'sexe':             req.sexe,
        'niveau':           'P1',
        'couleur':          'ROUGE',
        'action':           'Admission immédiate',
        'status':           'en_attente',
        'triage_id':        triage.id,
        'admission_directe': True,
    }
    add_patient(triage.id, 'P1', patient_data)

    await manager.broadcast({'event': 'admission_directe', 'patient': patient_data})

    return {
        'triage_id': triage.id,
        'niveau':    'P1',
        'confiance': 100,
        'action':    'Admission immédiate',
        'couleur':   'ROUGE',
    }
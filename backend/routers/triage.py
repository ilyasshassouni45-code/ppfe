# routers/triage.py

import os
import sys
import base64
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Body, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from sqlalchemy import case, func
from sqlalchemy.orm import Session

import database
import models

# Keep same import logic as your original file
BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from urgence.models.fusion.fusion import UrgenceClassifier

router = APIRouter()

active_connections = []


# ─────────────────────────────────────────────
# SCHEMAS
# ─────────────────────────────────────────────

class TriageRequest(BaseModel):
    nom: str
    symptomes: str
    age: Optional[int] = None
    sexe: Optional[str] = None
    antecedents: Optional[str] = None
    image_base64: Optional[str] = None


class AdmissionDirecteRequest(BaseModel):
    nom: str
    note: Optional[str] = None
    age: Optional[int] = None
    sexe: Optional[str] = None


# ─────────────────────────────────────────────
# REAL AI CLASSIFIER
# ─────────────────────────────────────────────

def get_classifier():
    if not hasattr(get_classifier, "_instance"):
        get_classifier._instance = UrgenceClassifier()
    return get_classifier._instance


def save_base64_image(image_base64: Optional[str]):
    if not image_base64:
        return None

    try:
        payload = image_base64

        # Accept both raw base64 and data:image/jpeg;base64,...
        if "," in payload:
            payload = payload.split(",", 1)[1]

        data = base64.b64decode(payload)

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        tmp.write(data)
        tmp.close()

        return tmp.name

    except Exception:
        return None


def format_triage(t: models.Triage):
    return {
        "id": t.id,
        "triage_id": t.id,
        "nom": t.patient_nom,
        "patient_nom": t.patient_nom,
        "age": t.patient_age,
        "sexe": t.patient_sexe,
        "niveau": t.niveau_urgence,
        "niveau_urgence": t.niveau_urgence,
        "confiance": t.confiance,
        "action": t.action,
        "couleur": t.couleur,
        "source": t.source,
        "keyword": t.keyword_p1,
        "status": t.status,
        "admission_directe": bool(t.admission_directe),
        "created_at": t.created_at.isoformat() if t.created_at else None,
        "scores": {
            "P1": t.score_p1,
            "P2": t.score_p2,
            "P3": t.score_p3,
            "P4": t.score_p4,
        },
    }


async def broadcast(message: dict):
    dead = []

    for websocket in active_connections:
        try:
            await websocket.send_json(message)
        except Exception:
            dead.append(websocket)

    for websocket in dead:
        if websocket in active_connections:
            active_connections.remove(websocket)


# ─────────────────────────────────────────────
# 1. CLASSIFIER — REAL MODEL, NOT RANDOM
# ─────────────────────────────────────────────

@router.post("/classifier")
async def classifier_patient(
    req: TriageRequest,
    db: Session = Depends(database.get_db)
):
    if not req.nom.strip():
        raise HTTPException(status_code=400, detail="Nom obligatoire")

    if not req.symptomes.strip():
        raise HTTPException(status_code=400, detail="Symptômes obligatoires")

    image_path = save_base64_image(req.image_base64)
    has_image = 1 if image_path else 0

    try:
        classifier = get_classifier()
        result = classifier.predict(req.symptomes, image_path)
    finally:
        if image_path and os.path.exists(image_path):
            os.unlink(image_path)

    scores = result.get("scores", {})

    triage = models.Triage(
        patient_nom=req.nom,
        patient_age=req.age,
        patient_sexe=req.sexe,
        symptomes=req.symptomes,
        niveau_urgence=result["niveau"],
        confiance=float(result["confiance"]),
        action=result["action"],
        couleur=result["couleur"],
        source=result["source"],
        keyword_p1=result.get("keyword"),
        has_image=has_image,
        score_p1=float(scores.get("P1", 0)),
        score_p2=float(scores.get("P2", 0)),
        score_p3=float(scores.get("P3", 0)),
        score_p4=float(scores.get("P4", 0)),
        status="en_attente",
        admission_directe=0,
    )

    db.add(triage)
    db.commit()
    db.refresh(triage)

    data = format_triage(triage)

    await broadcast({
        "event": "nouveau_patient",
        "patient": data,
    })

    return data


# ─────────────────────────────────────────────
# 2. ADMISSION DIRECTE P1
# ─────────────────────────────────────────────

@router.post("/admission-directe")
async def admission_directe(
    req: AdmissionDirecteRequest,
    db: Session = Depends(database.get_db)
):
    if not req.nom.strip():
        raise HTTPException(status_code=400, detail="Nom obligatoire")

    triage = models.Triage(
        patient_nom=req.nom,
        patient_age=req.age,
        patient_sexe=req.sexe,
        symptomes=req.note or "Admission directe P1",
        niveau_urgence="P1",
        confiance=100,
        action="Admission immédiate — urgence absolue",
        couleur="ROUGE",
        source="admission_directe",
        keyword_p1=None,
        has_image=0,
        score_p1=1,
        score_p2=0,
        score_p3=0,
        score_p4=0,
        status="en_attente",
        admission_directe=1,
    )

    db.add(triage)
    db.commit()
    db.refresh(triage)

    data = format_triage(triage)

    await broadcast({
        "event": "admission_directe",
        "patient": data,
    })

    return {
        "success": True,
        "triage_id": triage.id,
        "niveau": "P1",
        "confiance": 100,
        "action": triage.action,
        "couleur": triage.couleur,
    }


# ─────────────────────────────────────────────
# 3. QUEUE
# ─────────────────────────────────────────────

@router.get("/queue/")
def get_queue(db: Session = Depends(database.get_db)):
    priority_order = case(
        (models.Triage.niveau_urgence == "P1", 1),
        (models.Triage.niveau_urgence == "P2", 2),
        (models.Triage.niveau_urgence == "P3", 3),
        (models.Triage.niveau_urgence == "P4", 4),
        else_=5,
    )

    items = (
        db.query(models.Triage)
        .filter(models.Triage.status != "termine")
        .order_by(priority_order, models.Triage.created_at.asc())
        .all()
    )

    patients = [format_triage(t) for t in items]

    return {
        "total": len(patients),
        "patients": patients,
    }


# ─────────────────────────────────────────────
# 4. STATS
# ─────────────────────────────────────────────

@router.get("/queue/stats")
def get_queue_stats(db: Session = Depends(database.get_db)):
    total = (
        db.query(func.count(models.Triage.id))
        .filter(models.Triage.status != "termine")
        .scalar()
        or 0
    )

    p1 = (
        db.query(func.count(models.Triage.id))
        .filter(
            models.Triage.niveau_urgence == "P1",
            models.Triage.status != "termine",
        )
        .scalar()
        or 0
    )

    termine = (
        db.query(func.count(models.Triage.id))
        .filter(models.Triage.status == "termine")
        .scalar()
        or 0
    )

    return {
        "total": total,
        "p1": p1,
        "termine": termine,
    }


# ─────────────────────────────────────────────
# 5. UPDATE STATUS
# ─────────────────────────────────────────────

@router.put("/queue/{triage_id}/status")
async def update_status(
    triage_id: int,
    data: dict = Body(...),
    db: Session = Depends(database.get_db)
):
    status = data.get("status")

    if status not in ["en_attente", "en_cours", "termine"]:
        raise HTTPException(status_code=400, detail="Status invalide")

    triage = db.query(models.Triage).filter(models.Triage.id == triage_id).first()

    if not triage:
        raise HTTPException(status_code=404, detail="Triage introuvable")

    triage.status = status
    db.commit()
    db.refresh(triage)

    await broadcast({
        "event": "status_update",
        "triage_id": triage.id,
        "status": status,
    })

    return {
        "success": True,
        "triage_id": triage.id,
        "status": status,
    }


# ─────────────────────────────────────────────
# 6. WEBSOCKET
# ─────────────────────────────────────────────

@router.websocket("/ws/queue")
async def websocket_queue(websocket: WebSocket):
    await websocket.accept()
    active_connections.append(websocket)

    try:
        await websocket.send_json({"event": "connected"})

        while True:
            await websocket.receive_text()

    except WebSocketDisconnect:
        if websocket in active_connections:
            active_connections.remove(websocket)

    except Exception:
        if websocket in active_connections:
            active_connections.remove(websocket)
from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional, List
import database
import models
import random

router = APIRouter(prefix="/triage", tags=["Triage"])

# --- Mocking Safety Net ---
P1_KEYWORDS = [
    'decollement', 'decolllement', 'peau se detache', 'epiderme se detache',
    'stevens-johnson', 'lyell', 'ten ', 'epidermolyse', 'necrolyse',
    'purpura', 'petechie', 'vitropression',
    'necrose', 'necrose', 'fasciite', 'gangrene',
    'difficultes respiratoires', 'detresse respiratoire',
    'stridor', 'oedeme larynge', 'oedeme laryngee',
    'choc ', 'tension effondree', 'hypotension severe',
    'inconscient', 'perte de connaissance',
    'brulure 20', 'brulure 30', 'brulure etendue',
    'sepsis', 'septicemie', 'purpura fulminans',
    'anaphylaxie', 'anaphylactique',
]

def normalize_text(text: str) -> str:
    return (text.lower()
            .replace('é','e').replace('è','e').replace('ê','e')
            .replace('à','a').replace('â','a').replace('ô','o')
            .replace('û','u').replace('î','i').replace('ç','c')
            .replace("'",' ').replace('-',' '))

def safety_check(text: str):
    normalized = normalize_text(text)
    for kw in P1_KEYWORDS:
        if normalize_text(kw) in normalized:
            return {'is_p1': True, 'keyword': kw}
    return {'is_p1': False, 'keyword': None}

# --- WebSocket Manager ---
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            await connection.send_json(message)

manager = ConnectionManager()

# --- Schemas ---
class TriageRequest(BaseModel):
    nom: str
    age: Optional[int] = None
    sexe: Optional[str] = None
    symptomes: str
    antecedents: Optional[str] = None
    image_base64: Optional[str] = None

class AdmissionDirecteRequest(BaseModel):
    nom: str
    age: Optional[int] = None
    sexe: Optional[str] = None
    note: Optional[str] = None

# --- Endpoints ---

@router.post("/classifier")
async def classifier_patient(req: TriageRequest, db: Session = Depends(database.get_db)):
    # 1. Safety Net
    sn = safety_check(req.symptomes)
    if sn['is_p1']:
        niveau = 'P1'
        confiance = 100
        action = "Admission immédiate — urgence absolue"
        couleur = 'ROUGE'
        source = 'safety_net'
        keyword = sn['keyword']
        scores = {'P1': 1.0, 'P2': 0.0, 'P3': 0.0, 'P4': 0.0}
    else:
        # Mock AI Prediction
        niveau = random.choice(['P2', 'P3', 'P4'])
        confiance = random.randint(70, 95)
        actions = {
            'P2': 'Prise en charge < 10 minutes',
            'P3': 'Prise en charge < 60 minutes',
            'P4': "Consultation standard — salle d'attente",
        }
        colors = {'P2': 'ORANGE', 'P3': 'JAUNE', 'P4': 'VERT'}
        action = actions[niveau]
        couleur = colors[niveau]
        source = 'ai_mock'
        keyword = None
        
        # Random scores
        s_p1 = 0.0
        s_p2 = random.uniform(0, 0.3) if niveau != 'P2' else random.uniform(0.4, 0.7)
        s_p3 = random.uniform(0, 0.3) if niveau != 'P3' else random.uniform(0.4, 0.7)
        s_p4 = random.uniform(0, 0.3) if niveau != 'P4' else random.uniform(0.4, 0.7)
        total = s_p2 + s_p3 + s_p4
        scores = {'P1': s_p1, 'P2': s_p2/total, 'P3': s_p3/total, 'P4': s_p4/total}

    triage = models.Triage(
        patient_nom=req.nom,
        patient_age=req.age,
        patient_sexe=req.sexe,
        symptomes=req.symptomes,
        niveau_urgence=niveau,
        confiance=confiance,
        action=action,
        couleur=couleur,
        source=source,
        keyword_p1=keyword,
        has_image=1 if req.image_base64 else 0,
        score_p1=scores['P1'],
        score_p2=scores['P2'],
        score_p3=scores['P3'],
        score_p4=scores['P4'],
    )
    db.add(triage)
    db.commit()
    db.refresh(triage)

    patient_data = {
        'nom': req.nom,
        'age': req.age,
        'sexe': req.sexe,
        'niveau': niveau,
        'couleur': couleur,
        'action': action,
        'status': 'en_attente',
        'triage_id': triage.id,
    }
    await manager.broadcast({'event': 'nouveau_patient', 'patient': patient_data})

    return {
        'triage_id': triage.id,
        'niveau': niveau,
        'confiance': confiance,
        'action': action,
        'couleur': couleur,
        'source': source,
        'keyword': keyword,
        'scores': scores,
    }

@router.post("/admission-directe")
async def admission_directe(req: AdmissionDirecteRequest, db: Session = Depends(database.get_db)):
    triage = models.Triage(
        patient_nom=req.nom,
        patient_age=req.age,
        patient_sexe=req.sexe,
        symptomes=req.note or "Admission directe — état critique visible",
        niveau_urgence='P1',
        confiance=100,
        action="Admission immédiate — urgence absolue",
        couleur='ROUGE',
        source='admission_directe',
        admission_directe=1,
    )
    db.add(triage)
    db.commit()
    db.refresh(triage)

    patient_data = {
        'nom': req.nom,
        'age': req.age,
        'sexe': req.sexe,
        'niveau': 'P1',
        'couleur': 'ROUGE',
        'action': 'Admission immédiate',
        'status': 'en_attente',
        'triage_id': triage.id,
        'admission_directe': True,
    }
    await manager.broadcast({'event': 'admission_directe', 'patient': patient_data})

    return {
        'triage_id': triage.id,
        'niveau': 'P1',
        'confiance': 100,
        'action': 'Admission immédiate',
        'couleur': 'ROUGE',
    }

@router.get("/queue/")
async def get_queue(db: Session = Depends(database.get_db)):
    triages = db.query(models.Triage).order_by(models.Triage.created_at).all()
    patients = []
    for t in triages:
        patients.append({
            'triage_id': t.id,
            'nom': t.patient_nom,
            'niveau': t.niveau_urgence,
            'action': t.action,
            'status': t.status,
            'admission_directe': bool(t.admission_directe)
        })
    return {
        'patients': patients,
        'total': len(patients)
    }

@router.get("/queue/stats")
async def get_queue_stats(db: Session = Depends(database.get_db)):
    triages = db.query(models.Triage).all()
    return {
        'total': len(triages),
        'p1': len([t for t in triages if t.niveau_urgence == 'P1']),
        'termine': len([t for t in triages if t.status == 'termine']),
    }

@router.put("/queue/{triage_id}/status")
async def update_status(triage_id: int, data: dict, db: Session = Depends(database.get_db)):
    status = data.get("status")
    triage = db.query(models.Triage).filter(models.Triage.id == triage_id).first()
    if not triage:
        raise HTTPException(status_code=404, detail="Triage not found")
    triage.status = status
    db.commit()
    await manager.broadcast({'event': 'status_update', 'triage_id': triage_id, 'status': status})
    return {"status": "updated"}

@router.websocket("/ws/queue")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

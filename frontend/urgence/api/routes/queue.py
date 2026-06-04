import sys
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

sys.path.insert(0, r"C:\Users\ismai\OneDrive\Desktop\urgence_system")

from api.database import get_db
from api.models_db import Triage
from api.websocket_manager import manager
from queue_manager import get_queue, remove_patient, update_status

router = APIRouter(prefix="/queue", tags=["Queue"])


class StatusUpdate(BaseModel):
    status: str


@router.get("/")
def get_queue_state():
    patients = get_queue()
    return {'total': len(patients), 'patients': patients}


@router.put("/{triage_id}/status")
async def update_patient_status(
    triage_id: int,
    body: StatusUpdate,
    db: Session = Depends(get_db)
):
    triage = db.query(Triage).filter(Triage.id == triage_id).first()
    if not triage:
        raise HTTPException(status_code=404, detail="Patient introuvable")

    triage.status = body.status
    db.commit()

    update_status(triage_id, body.status)

    if body.status == 'termine':
        remove_patient(triage_id)

    await manager.broadcast({
        'event':     'status_update',
        'triage_id': triage_id,
        'status':    body.status,
    })

    return {'triage_id': triage_id, 'status': body.status}


@router.get("/stats")
def get_stats(db: Session = Depends(get_db)):
    from sqlalchemy import func
    total = db.query(func.count(Triage.id)).scalar()
    by_n  = db.query(Triage.niveau_urgence, func.count(Triage.id)) \
              .group_by(Triage.niveau_urgence).all()
    return {
        'total':     total,
        'by_niveau': {n: c for n, c in by_n},
    }
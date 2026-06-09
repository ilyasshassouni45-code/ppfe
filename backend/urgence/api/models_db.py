from sqlalchemy import Column, Integer, String, Float, Boolean, Text, DateTime
from sqlalchemy.sql import func
from api.database import Base


class Triage(Base):
    __tablename__ = "triages"

    id                = Column(Integer, primary_key=True, index=True)
    patient_nom       = Column(String(100))
    patient_age       = Column(Integer,     nullable=True)
    patient_sexe      = Column(String(1),   nullable=True)
    symptomes         = Column(Text)
    niveau_urgence    = Column(String(2))
    confiance         = Column(Integer)
    action            = Column(Text)
    couleur           = Column(String(10))
    source            = Column(String(20))
    keyword_p1        = Column(String(100), nullable=True)
    has_image         = Column(Boolean,     default=False)
    score_p1          = Column(Float,       default=0.0)
    score_p2          = Column(Float,       default=0.0)
    score_p3          = Column(Float,       default=0.0)
    score_p4          = Column(Float,       default=0.0)
    admission_directe = Column(Boolean,     default=False)
    status            = Column(String(20),  default='en_attente')
    created_at        = Column(DateTime,    default=func.now())
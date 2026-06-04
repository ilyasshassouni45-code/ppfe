import sys
sys.path.insert(0, r"C:\Users\ismai\OneDrive\Desktop\urgence_system")

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from api.database import engine
from api.models_db import Base
from api.routes.triage import router as triage_router, get_classifier
from api.routes.queue  import router as queue_router
from api.websocket_manager import manager


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Charger le classifier au démarrage
    print("Initialisation du classifier AI...")
    get_classifier()
    print("✅ Classifier prêt\n")
    yield


app = FastAPI(
    title="Système Urgence Dermatologie",
    description="API triage intelligent",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(triage_router)
app.include_router(queue_router)


@app.websocket("/ws/queue")
async def websocket_queue(ws: WebSocket):
    await manager.connect(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(ws)


@app.get("/")
def root():
    return {"message": "Système Urgence Dermatologie — API opérationnelle", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "ok"}
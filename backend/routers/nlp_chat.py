import os
from fastapi import APIRouter
from pydantic import BaseModel
from chatbot.bot import DermaFlowBot

# Path to knowledge base
KB_PATH = os.path.join(
    os.path.dirname(__file__), 
    '..', 
    'data', 
    'knowledge_base.json'
)

router = APIRouter(prefix="/nlp-chat", tags=["NLP Chatbot"])

# Single bot instance — loaded once at startup
bot = DermaFlowBot(kb_path=KB_PATH)


class MessageIn(BaseModel):
    message: str


@router.post("/")
def chat(payload: MessageIn):
    """
    Receive user message → NLP processing → return response.
    POST /nlp-chat/
    Body: { "message": "..." }
    """
    return bot.respond(payload.message)


@router.get("/health")
def chatbot_health():
    """
    Confirms bot is loaded and shows KB size.
    GET /nlp-chat/health
    """
    return {
        "status":  "ok",
        "kb_size": len(bot.matcher.knowledge_base)
    }
from chatbot.matcher import Matcher

FALLBACK = [
    "Je ne suis pas sûr de comprendre. Pouvez-vous reformuler ?",
    "Je n'ai pas d'information sur ce sujet. Appelez-nous au 05XX-XXXXXX.",
    "Pour cette question, notre équipe peut mieux vous aider directement.",
]

class DermaFlowBot:
    def __init__(self, kb_path: str):
        self.matcher      = Matcher(kb_path)
        self._fallback_i  = 0

    def respond(self, user_input: str) -> dict:
        if not user_input or not user_input.strip():
            return {"answer": "Bonjour ! Comment puis-je vous aider ?",
                    "score": 1.0, "matched": None}

        entry, score = self.matcher.find_best_match(user_input)

        if entry:
            return {"answer":   entry["answer"],
                    "score":    round(score, 3),
                    "matched":  entry["question"]}

        fallback = FALLBACK[self._fallback_i % len(FALLBACK)]
        self._fallback_i += 1
        return {"answer": fallback, "score": round(score, 3), "matched": None}
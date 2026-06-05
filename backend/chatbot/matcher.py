import json
import math
from chatbot.preprocessor import tokenize

class Matcher:
    def __init__(self, kb_path: str):
        with open(kb_path, 'r', encoding='utf-8') as f:
            self.knowledge_base = json.load(f)

        self.documents = []
        for entry in self.knowledge_base:
            combined = " ".join(entry["tags"]) + " " + entry["question"]
            self.documents.append(tokenize(combined))

        self.idf = self._compute_idf()

    def _compute_idf(self) -> dict:
        N = len(self.documents)
        df = {}
        for doc in self.documents:
            for token in set(doc):
                df[token] = df.get(token, 0) + 1
        return {
            token: math.log((N + 1) / (freq + 1)) + 1
            for token, freq in df.items()
        }

    def _tfidf_vector(self, tokens: list) -> dict:
        tf = {}
        for token in tokens:
            tf[token] = tf.get(token, 0) + 1
        total = len(tokens) if tokens else 1
        return {
            token: (count / total) * self.idf.get(token, math.log(2) + 1)
            for token, count in tf.items()
        }

    def _cosine_similarity(self, vec1: dict, vec2: dict) -> float:
        dot   = sum(vec1.get(t, 0) * vec2.get(t, 0) for t in vec2)
        norm1 = math.sqrt(sum(v ** 2 for v in vec1.values()))
        norm2 = math.sqrt(sum(v ** 2 for v in vec2.values()))
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot / (norm1 * norm2)

    def find_best_match(self, user_input: str, threshold: float = 0.15):
        user_tokens = tokenize(user_input)
        if not user_tokens:
            return None, 0.0

        user_vec   = self._tfidf_vector(user_tokens)
        best_score = 0.0
        best_entry = None

        for i, doc_tokens in enumerate(self.documents):
            score = self._cosine_similarity(user_vec, self._tfidf_vector(doc_tokens))
            if score > best_score:
                best_score = score
                best_entry = self.knowledge_base[i]

        if best_score >= threshold:
            return best_entry, best_score
        return None, best_score
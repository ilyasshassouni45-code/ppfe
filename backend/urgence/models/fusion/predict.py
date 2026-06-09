import sys
sys.path.insert(0, r"C:\Users\ismai\OneDrive\Desktop\urgence_system")

from models.fusion.fusion import UrgenceClassifier

COLORS_CLI = {
    'P1': '\033[91m', 'P2': '\033[93m',
    'P3': '\033[33m', 'P4': '\033[92m',
    'RESET': '\033[0m',
}

_classifier = None


def get_classifier():
    global _classifier
    if _classifier is None:
        _classifier = UrgenceClassifier()
    return _classifier


def predict(symptomes: str, image_path: str = None) -> dict:
    result = get_classifier().predict(symptomes, image_path)

    c = COLORS_CLI.get(result['niveau'], '')
    r = COLORS_CLI['RESET']

    print("\n" + "="*52)
    print(f"  NIVEAU  : {c}● {result['niveau']} — {result['couleur']}{r}")
    print(f"  Confiance: {result['confiance']}%")
    print(f"  Action  : {result['action']}")
    print(f"  Source  : {result['source']}")
    if result.get('keyword'):
        print(f"  Mot clé : '{result['keyword']}'")
    print("\n  Scores:")
    for cls, score in result['scores'].items():
        bar = '█' * int(score * 20)
        print(f"    {cls}: {bar:<20} {score*100:.1f}%")
    print("="*52 + "\n")

    return result


if __name__ == '__main__':
    tests = [
        ("P1 Safety Net", "Decollement cutane etendu, difficultes respiratoires", None),
        ("P2 AI",         "Erythrodermie etendue avec fievre a 39C depuis 48h",   None),
        ("P3 AI",         "Zona intercostal avec douleurs neuropathiques intenses", None),
        ("P4 AI",         "Acne inflammatoire moderee sur le visage",              None),
    ]
    print("━━━ Test Système Fusion ━━━")
    for label, symptomes, img in tests:
        print(f"\n[{label}]")
        predict(symptomes, img)
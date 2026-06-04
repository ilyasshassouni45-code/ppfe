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


def _normalize(text: str) -> str:
    return (text.lower()
            .replace('é','e').replace('è','e').replace('ê','e')
            .replace('à','a').replace('â','a').replace('ô','o')
            .replace('û','u').replace('î','i').replace('ç','c')
            .replace("'",' ').replace('-',' '))


def check(text: str) -> dict:
    normalized = _normalize(text)
    for kw in P1_KEYWORDS:
        if _normalize(kw) in normalized:
            return {'is_p1': True, 'keyword': kw}
    return {'is_p1': False, 'keyword': None}


if __name__ == '__main__':
    tests = [
        ("Patient avec decollement cutane etendu", True),
        ("Purpura ne s'effacant pas a la vitropression", True),
        ("Acne legere sur le visage", False),
        ("Eczema modere avec prurit", False),
        ("Stridor avec difficultes respiratoires", True),
        ("Psoriasis en plaques stable", False),
    ]
    print("━━━ Test Safety Net ━━━\n")
    all_ok = True
    for text, expected in tests:
        result = check(text)
        ok = result['is_p1'] == expected
        if not ok:
            all_ok = False
        status = "✅" if ok else "❌"
        kw = f"→ '{result['keyword']}'" if result['is_p1'] else ""
        print(f"  {status} [{('P1' if result['is_p1'] else 'OK'):2s}] {text[:50]} {kw}")
    print(f"\n{'✅ Tous les tests passés' if all_ok else '❌ Certains tests échoués'}")
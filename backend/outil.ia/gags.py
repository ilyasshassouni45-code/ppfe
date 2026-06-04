def calculer_gags(forehead, right_cheek, left_cheek, nose, chin):
    factors = {
        'forehead':    2,
        'right_cheek': 2,
        'left_cheek':  2,
        'nose':        1,
        'chin':        1
    }

    score = (forehead    * factors['forehead']    +
             right_cheek * factors['right_cheek'] +
             left_cheek  * factors['left_cheek']  +
             nose        * factors['nose']        +
             chin        * factors['chin'])

    if score == 0:
        severite  = "Clear"
        traitement = "Aucun traitement necessaire"
    elif score <= 18:
        severite  = "Mild"
        traitement = "Traitement topique"
    elif score <= 30:
        severite  = "Moderate"
        traitement = "Consultation recommandee"
    elif score <= 38:
        severite  = "Severe"
        traitement = "Traitement medical urgent"
    else:
        severite  = "Very Severe"
        traitement = "Traitement intensif immediat"

    return {
        'score':      score,
        'severite':   severite,
        'traitement': traitement
    }


def saisir_et_calculer():
    zones = {
        'forehead':    'Forehead',
        'right_cheek': 'Right cheek',
        'left_cheek':  'Left cheek',
        'nose':        'Nose',
        'chin':        'Chin'
    }

    grades = {}
    for key, label in zones.items():
        while True:
            try:
                val = int(input(f"{label} [0-4]: "))
                if 0 <= val <= 4:
                    grades[key] = val
                    break
            except ValueError:
                pass

    resultat = calculer_gags(**grades)

    print("\n" + "="*45)
    print("RESULTAT GAGS:")
    print(f"Score      : {resultat['score']}")
    print(f"Severite   : {resultat['severite']}")
    print(f"Traitement : {resultat['traitement']}")
    print("="*45 + "\n")


if __name__ == "__main__":
    saisir_et_calculer()
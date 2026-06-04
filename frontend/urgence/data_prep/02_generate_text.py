import os
import random
import pandas as pd

random.seed(42)
os.makedirs(
    os.path.join(r"C:\Users\ismai\OneDrive\Desktop\urgence_system", "data", "synthetic"),
    exist_ok=True
)

SCENARIOS = {
    'P1': {
        'motifs': [
            "decollement cutane etendu sur une large surface corporelle",
            "purpura petechial ne s effacant pas a la vitropression",
            "brulures chimiques couvrant plus de 20% de la surface cutanee",
            "necrose cutanee a progression rapide",
            "erytheme bulleux generalise avec decollement epidermique",
            "reaction anaphylactique severe avec urticaire geante",
            "syndrome de Stevens-Johnson avec atteinte des muqueuses",
            "syndrome de Lyell avec decollement cutane spontane",
            "sepsis cutane avec purpura fulminans",
        ],
        'signes': [
            "difficultes respiratoires importantes",
            "oedeme larynge avec stridor audible",
            "fievre a 40C avec frissons intenses",
            "etat hemodynamique instable",
            "conscience alteree",
            "tension arterielle effondree",
            "atteinte severe des muqueuses buccales et oculaires",
        ],
        'duree': [
            "apparu brutalement il y a moins d une heure",
            "evolution fulminante depuis 2 heures",
            "debut brutal depuis ce matin",
            "aggravation tres rapide en moins de 3 heures",
        ],
        'localisation': [
            "etendu au tronc et aux quatre membres",
            "generalise sur l ensemble du corps",
            "predominant au visage avec extension rapide",
            "touchant le visage et les voies aeriennes",
        ],
        'contexte': [
            "sous traitement antibiotique depuis 5 jours",
            "introduction d un nouveau medicament il y a 1 semaine",
            "exposition a un produit chimique industriel",
            "antecedent d allergie medicamenteuse connue",
            "aucun antecedent notable",
        ],
    },
    'P2': {
        'motifs': [
            "erythrodermie touchant plus de 80% de la surface corporelle",
            "suspicion de syndrome DRESS avec eruption etendue",
            "brulures thermiques de 10 a 20% de la surface cutanee",
            "cellulite extensive du membre inferieur avec signes generaux",
            "zona ophtalmique avec atteinte corneenne suspectee",
            "fasciite necrosante debutante suspectee",
            "pemphigoide bulleuse etendue non controlee",
            "vascularite cutanee avec atteinte systemique",
            "erysipele etendu resistant au traitement ambulatoire",
        ],
        'signes': [
            "fievre a 39C persistante",
            "adenopathies cervicales et axillaires",
            "prurit intense resistant aux antihistaminiques",
            "douleur cotee a 7 sur 10",
            "vesicules confluentes en nappe",
            "oedeme important du membre atteint",
            "alteration marquee de l etat general",
        ],
        'duree': [
            "evolution progressive depuis 24 heures",
            "aggravation nette depuis 48 heures",
            "persistance depuis 3 jours malgre traitement",
            "recidive severe depuis hier soir",
        ],
        'localisation': [
            "predominant sur le tronc avec extension aux membres",
            "touchant les membres superieurs et inferieurs",
            "au niveau du visage avec extension peri-orbitaire",
            "sur tout le membre inferieur gauche",
        ],
        'contexte': [
            "sous corticotherapie orale depuis 3 semaines",
            "patient diabetique avec glycemie desequilibree",
            "immunodeprime sous chimiotherapie",
            "antibiotique deja instaure sans amelioration",
            "premier episode sans antecedent particulier",
        ],
    },
    'P3': {
        'motifs': [
            "zona intercostal avec douleurs neuropathiques intenses",
            "urticaire aigue etendue sans signes anaphylactiques",
            "eczema surinfecte localise avec suintement",
            "abces cutane fluctuant necessitant drainage",
            "dermatite de contact allergique severe",
            "psoriasis en pustules localise",
            "herpes labial etendu en recidive",
            "folliculite dissequante du cuir chevelu",
            "lichen plan erosif buccal douloureux",
        ],
        'signes': [
            "prurit modere a intense",
            "legere fievre a 38C",
            "rougeur localisee avec chaleur locale",
            "suintement sereux localise",
            "douleur moderee cotee a 4 sur 10",
            "petits ganglions palpables localement",
        ],
        'duree': [
            "evolution depuis 3 a 5 jours",
            "present depuis une semaine avec aggravation recente",
            "persistance depuis 10 jours sans amelioration",
            "aggravation depuis 4 jours malgre traitement local",
        ],
        'localisation': [
            "au niveau du thorax droit",
            "sur les avant-bras et les mains",
            "au niveau du cou et de la nuque",
            "sur la jambe droite",
            "au niveau du cuir chevelu",
        ],
        'contexte': [
            "traitement topique insuffisant",
            "premier episode chez un adulte jeune",
            "terrain atopique connu",
            "contact recent avec un allergene identifie",
            "retour de voyage recent",
        ],
    },
    'P4': {
        'motifs': [
            "acne inflammatoire moderee avec quelques lesions nodulaires",
            "eczema atopique en legere poussee",
            "psoriasis en plaques chronique stable",
            "lentigos et taches pigmentees sans evolution",
            "consultation de suivi dermatologique de routine",
            "verrues vulgaires multiples aux mains",
            "seborrhee du cuir chevelu avec pellicules abondantes",
            "cheloide post-cicatricielle inesthetique",
            "onychomycose debutante des orteils",
            "rosacee faciale stable sous traitement",
            "xerose cutanee hivernale importante",
            "naevus melanocytaire stable a surveiller",
            "alopecie androgenetique debutante",
        ],
        'signes': [
            "etat general parfaitement conserve",
            "apyretique",
            "prurit leger et intermittent",
            "gene esthetique sans douleur",
            "legere desquamation localisee",
            "aucun signe systemique associe",
        ],
        'duree': [
            "evolution chronique depuis plusieurs semaines",
            "probleme connu depuis plusieurs mois",
            "stable depuis longtemps",
            "legere aggravation depuis 2 a 3 semaines",
        ],
        'localisation': [
            "sur le visage et le front",
            "dans le dos et les epaules",
            "aux membres inferieurs",
            "sur le cuir chevelu",
            "aux ongles des deux pieds",
            "sur les mains et les poignets",
        ],
        'contexte': [
            "renouvellement d ordonnance habituelle",
            "premier episode chez un adolescent",
            "suivi regulier sans complication",
            "changement de produits cosmetiques recent",
            "stress signale par le patient",
        ],
    },
}

TEMPLATES = [
    "Motif: {motif} {localisation}. Duree: {duree}. Signes: {signe}. Contexte: {contexte}.",
    "Patient consulte pour {motif} {localisation}, {duree}. On note {signe}. {contexte}.",
    "{motif} {localisation} ({duree}). Le patient rapporte {signe}. {contexte}.",
    "Consultation pour {motif} {localisation}. Evolution: {duree}. Clinique: {signe}. {contexte}.",
    "Motif de venue: {motif} {localisation}. Depuis: {duree}. Associe a {signe}. {contexte}.",
]

COUNTS = {'P1': 500, 'P2': 700, 'P3': 900, 'P4': 1200}

rows = []
for level, n in COUNTS.items():
    db = SCENARIOS[level]
    for _ in range(n):
        rows.append({
            'texte': random.choice(TEMPLATES).format(
                motif=random.choice(db['motifs']),
                signe=random.choice(db['signes']),
                duree=random.choice(db['duree']),
                localisation=random.choice(db['localisation']),
                contexte=random.choice(db['contexte']),
            ),
            'urgence':     level,
            'urgence_num': int(level[1]),
        })

df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
out = os.path.join(
    r"C:\Users\ismai\OneDrive\Desktop\urgence_system",
    "data", "synthetic", "text_scenarios.csv"
)
df.to_csv(out, index=False)

print("Distribution:")
print(df['urgence'].value_counts().sort_index().to_string())
print(f"\n✅ {len(df)} scenarios generes")
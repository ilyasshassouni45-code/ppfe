import string

STOPWORDS_FR = {
    "le","la","les","un","une","des","de","du","et","en","est","au","aux",
    "ce","se","sa","son","ses","je","tu","il","elle","nous","vous","ils",
    "que","qui","quoi","ou","comment","quand","pourquoi","quel","quelle",
    "par","sur","dans","avec","pour","pas","ne","plus","tres","bien","on",
    "y","me","te","lui","leur","si","mais","donc","car","or","ni","car"
}

def normalize(text: str) -> str:
    text = text.lower().strip()
    accents = {
        'é':'e','è':'e','ê':'e','ë':'e',
        'à':'a','â':'a','ä':'a',
        'ù':'u','û':'u','ü':'u',
        'ô':'o','ö':'o',
        'î':'i','ï':'i',
        'ç':'c'
    }
    for src, dst in accents.items():
        text = text.replace(src, dst)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

def tokenize(text: str) -> list:
    tokens = normalize(text).split()
    return [t for t in tokens if t not in STOPWORDS_FR and len(t) > 2]
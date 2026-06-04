import os
from ultralytics import YOLO
from tkinter import Tk, filedialog
import numpy as np

def selectionner_et_analyser():
    root = Tk()
    root.withdraw()
    root.attributes('-topmost', True)

    chemin_image = filedialog.askopenfilename(
        title="Sélectionner une image de lésion cutanée",
        filetypes=[("Images", "*.jpg *.jpeg *.png *.webp")]
    )

    if not chemin_image:
        return

    model_path = "skin_lesion_classifier.pt"
    if not os.path.exists(model_path):
        return

    model = YOLO(model_path)
    results = model(chemin_image)[0]

    probs = results.probs.data.cpu().numpy()
    names = results.names

    descriptions = {
        'Dermatitis': 'Inflammation cutanée - Traitement topique',
        'Eczema':     'Eczéma - Crème hydratante + corticoïdes',
        'Rosacea':    'Rosacée - Traitement médical recommandé',
        'Normal':     'Peau saine - Aucun traitement'
    }

    top3_idx = [i for i in np.argsort(probs)[::-1]
                if names[i] != 'Psoriasis'][:3]

    print("\n" + "="*50)
    print("RÉSULTAT SKIN LESION TRIAGE:")
    print(f"Image: {os.path.basename(chemin_image)}")
    print("\nTop-3 diagnostics:")
    for i, idx in enumerate(top3_idx, 1):
        cls = names[idx]
        conf = probs[idx] * 100
        desc = descriptions.get(cls, '')
        print(f"  #{i} {cls}: {conf:.1f}% - {desc}")
    print("="*50 + "\n")

if __name__ == "__main__":
    selectionner_et_analyser()
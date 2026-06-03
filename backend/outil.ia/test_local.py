import os
from ultralytics import YOLO
from tkinter import Tk, filedialog
from PIL import Image

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

    model_path = "melanoma_classifier.pt"
    if not os.path.exists(model_path):
        return

    model = YOLO(model_path)
    results = model(chemin_image)[0]

    top_class_index = results.probs.top1
    top_class_name = results.names[top_class_index]
    confidence = float(results.probs.top1conf) * 100

    risk_levels = {
        'Benign': 'Faible risque - Surveillance',
        'Malignant': 'RISQUE ELEVE - Biopsy urgente'
    }
    risque = risk_levels.get(top_class_name, 'Inconnu')

    print("\n" + "="*40)
    print("RESULTAT DE L'ANALYSE:")
    print(f"Image     : {os.path.basename(chemin_image)}")
    print(f"Classe    : {top_class_name}")
    print(f"Confiance : {confidence:.1f}%")
    print(f"Risque    : {risque}")
    print("="*40 + "\n")

if __name__ == "__main__":
    selectionner_et_analyser()
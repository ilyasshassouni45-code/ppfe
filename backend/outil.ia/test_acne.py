import os
from ultralytics import YOLO
from tkinter import Tk, filedialog

def selectionner_et_analyser():
    root = Tk()
    root.withdraw()
    root.attributes('-topmost', True)

    chemin_image = filedialog.askopenfilename(
        title="Sélectionner une image acné",
        filetypes=[("Images", "*.jpg *.jpeg *.png *.webp")]
    )

    if not chemin_image:
        return

    model_path = "acne_classifier.pt"
    if not os.path.exists(model_path):
        return

    model = YOLO(model_path)
    results = model(chemin_image)[0]

    top_class_index = results.probs.top1
    top_class_name = results.names[top_class_index]
    confidence = float(results.probs.top1conf) * 100

    severity_levels = {
        'Clear':    'Peau saine - Aucun traitement',
        'Mild':     'Acne legere - Traitement topique',
        'Moderate': 'Acne moderee - Consultation recommandee',
        'Severe':   'Acne severe - Traitement urgent'
    }
    niveau = severity_levels.get(top_class_name, 'Inconnu')

    print("\n" + "="*45)
    print("RESULTAT CLASSIFICATION ACNE:")
    print(f"Image     : {os.path.basename(chemin_image)}")
    print(f"Severite  : {top_class_name}")
    print(f"Confiance : {confidence:.1f}%")
    print(f"Niveau    : {niveau}")
    print("="*45 + "\n")

if __name__ == "__main__":
    selectionner_et_analyser()
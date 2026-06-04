import os
import random
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter, UnidentifiedImageError
from tqdm import tqdm

random.seed(42)

BASE     = r"C:\Users\ismai\OneDrive\Desktop\urgence_system"
PROC_DIR = os.path.join(BASE, "data", "processed", "images")
IMG_SIZE = (224, 224)
TARGET   = {'P1': 1500, 'P2': 2500, 'P3': None, 'P4': None}


def augment(img):
    return [
        img.transpose(Image.FLIP_LEFT_RIGHT),
        img.rotate(random.uniform(-15, 15), fillcolor=(0, 0, 0)),
        ImageEnhance.Brightness(img).enhance(random.uniform(0.8, 1.2)),
        ImageEnhance.Contrast(img).enhance(random.uniform(0.8, 1.2)),
        img.filter(ImageFilter.GaussianBlur(radius=0.5)),
        ImageEnhance.Color(img).enhance(random.uniform(0.8, 1.2)),
    ]


def process(urgence, target):
    folder = os.path.join(PROC_DIR, urgence)
    exts   = {'.jpg', '.jpeg', '.png', '.bmp'}
    files  = [f for f in os.listdir(folder)
               if Path(f).suffix.lower() in exts]

    if not files:
        print(f"  ⚠️  {urgence}: dossier vide — skip")
        return

    # Resize
    errors = 0
    for fname in tqdm(files, desc=f"  Resize {urgence}"):
        path = os.path.join(folder, fname)
        base = os.path.splitext(fname)[0]
        dst  = os.path.join(folder, f"{base}.jpg")
        try:
            img = Image.open(path).convert('RGB').resize(IMG_SIZE, Image.LANCZOS)
            img.save(dst, 'JPEG', quality=90)
            if path != dst and os.path.exists(path):
                os.remove(path)
        except (UnidentifiedImageError, Exception):
            errors += 1
            if os.path.exists(path):
                try:
                    os.remove(path)
                except Exception:
                    pass

    current = [f for f in os.listdir(folder) if f.endswith('.jpg')]
    print(f"  {urgence}: {len(current)} images | erreurs ignorées: {errors}")

    if target is None or len(current) >= target:
        return

    # Augmentation
    needed = target - len(current)
    done, cycle = 0, 0
    print(f"  Augmentation {urgence}: {needed} images a generer...")

    while done < needed:
        if not current:
            break
        fname = current[cycle % len(current)]
        path  = os.path.join(folder, fname)
        try:
            img = Image.open(path).convert('RGB')
            for aug_img in augment(img):
                if done >= needed:
                    break
                base = os.path.splitext(fname)[0]
                aug_img.save(
                    os.path.join(folder, f"{base}_aug{done:05d}.jpg"),
                    'JPEG', quality=85
                )
                done += 1
        except Exception:
            pass
        cycle += 1

    total = len([f for f in os.listdir(folder) if f.endswith('.jpg')])
    print(f"  {urgence}: {total} images apres augmentation ✅")


def rebuild_meta():
    rows = []
    for u in ['P1', 'P2', 'P3', 'P4']:
        d = os.path.join(PROC_DIR, u)
        if not os.path.exists(d):
            continue
        for f in os.listdir(d):
            if f.endswith('.jpg'):
                rows.append({
                    'image_path':  os.path.join(d, f),
                    'urgence':     u,
                    'urgence_num': int(u[1]),
                    'augmented':   '_aug' in f,
                })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(BASE, "data", "processed", "metadata_processed.csv"),
              index=False)
    return df


if __name__ == "__main__":
    from pathlib import Path

    print("━━━ Preprocessing ━━━\n")
    for u, t in TARGET.items():
        process(u, t)

    df = rebuild_meta()
    print("\n📊 Distribution finale:")
    print(df['urgence'].value_counts().sort_index().to_string())
    print(f"\n✅ Total: {len(df)} images")
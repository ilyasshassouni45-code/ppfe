import os
import shutil
import pandas as pd
from pathlib import Path
from tqdm import tqdm

BASE = r"C:\Users\ismai\OneDrive\Desktop\urgence_system"

DATASETS = {
    'ham10000': os.path.join(BASE, "data", "raw", "ham10000"),
    'isic':     os.path.join(BASE, "data", "raw", "isic"),
    'ph2':      os.path.join(BASE, "data", "raw", "ph2"),
}

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp'}

def find_files(folder, exts=None, name_contains=None):
    results = []
    for root, _, files in os.walk(folder):
        for f in files:
            ext = Path(f).suffix.lower()
            if exts and ext not in exts:
                continue
            if name_contains and name_contains.lower() not in f.lower():
                continue
            results.append(os.path.join(root, f))
    return results

def organize_dataset(name, folder):
    out = os.path.join(folder, "images_all")
    os.makedirs(out, exist_ok=True)
    imgs = find_files(folder, exts=IMG_EXTS)
    imgs = [i for i in imgs if 'images_all' not in i]
    copied = 0
    for src in tqdm(imgs, desc=f"  {name}"):
        dst = os.path.join(out, Path(src).name)
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
            copied += 1
    total = len(os.listdir(out))
    print(f"  ✅ {name}: {total} images dans images_all/")
    return total

def find_metadata(folder, dataset):
    csvs  = find_files(folder, exts={'.csv'})
    xlsxs = find_files(folder, exts={'.xlsx'})
    if dataset == 'ham10000':
        for c in csvs:
            if 'metadata' in c.lower() or 'ham' in c.lower():
                return c, 'csv'
        if csvs:
            return csvs[0], 'csv'
    elif dataset == 'isic':
        for c in csvs:
            if 'ground' in c.lower() or 'truth' in c.lower() or 'label' in c.lower():
                return c, 'csv'
        if csvs:
            return csvs[0], 'csv'
    elif dataset == 'ph2':
        if xlsxs:
            return xlsxs[0], 'xlsx'
        if csvs:
            return csvs[0], 'csv'
    return None, None

if __name__ == "__main__":
    print("━━━ Organisation datasets ━━━\n")
    summary = {}
    for name, folder in DATASETS.items():
        if not os.path.exists(folder):
            print(f"⚠️  {name}: dossier introuvable — {folder}")
            continue
        print(f"[{name.upper()}]")
        n = organize_dataset(name, folder)
        summary[name] = n
        meta_path, meta_type = find_metadata(folder, name)
        if meta_path:
            ext = '.csv' if meta_type == 'csv' else '.xlsx'
            dst = os.path.join(folder, f"metadata{ext}")
            if meta_path != dst:
                shutil.copy2(meta_path, dst)
            print(f"  ✅ Metadata: {Path(meta_path).name}")
            if meta_type == 'csv':
                df = pd.read_csv(meta_path, nrows=3)
            else:
                df = pd.read_excel(meta_path, nrows=3)
            print(f"  Colonnes: {list(df.columns)[:8]}")
        else:
            print(f"  ⚠️  Aucune metadata trouvée")
        print()
    print("━━━ Résumé ━━━")
    for name, n in summary.items():
        print(f"  {name:12s}: {n} images")
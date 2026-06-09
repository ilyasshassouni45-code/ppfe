import os
import shutil
import pandas as pd
from pathlib import Path
from tqdm import tqdm

BASE     = r"C:\Users\ismai\OneDrive\Desktop\urgence_system"
OUT_DIR  = os.path.join(BASE, "data", "processed", "images")

for p in ['P1', 'P2', 'P3', 'P4']:
    os.makedirs(os.path.join(OUT_DIR, p), exist_ok=True)

MAPPING_HAM = {
    'mel':   'P2', 'vasc':  'P2',
    'akiec': 'P3', 'bcc':   'P3',
    'bkl':   'P4', 'df':    'P4', 'nv': 'P4',
}

MAPPING_ISIC = {
    'MEL':  'P2', 'VASC': 'P2', 'SCC': 'P2',
    'BCC':  'P3', 'AK':   'P3',
    'BKL':  'P4', 'DF':   'P4', 'NV':  'P4',
}

MAPPING_PH2 = {1: 'P4', 2: 'P3', 3: 'P2'}

records   = []
not_found = 0


def copy_image(src, urgence, prefix):
    global not_found
    if not os.path.exists(src):
        not_found += 1
        return False
    stem = Path(src).stem
    ext  = Path(src).suffix.lower()
    dst  = os.path.join(OUT_DIR, urgence, f"{prefix}_{stem}{ext}")
    if not os.path.exists(dst):
        shutil.copy2(src, dst)
    records.append({
        'image_path':  dst,
        'urgence':     urgence,
        'urgence_num': int(urgence[1]),
        'source':      prefix,
        'augmented':   False,
    })
    return True


# ── HAM10000 ──────────────────────────────────────────────────────────────────

def relabel_ham():
    csv_path = os.path.join(BASE, "data", "raw", "ham10000", "HAM10000_metadata.csv")
    img_dir  = os.path.join(BASE, "data", "raw", "ham10000", "images")

    if not os.path.exists(csv_path):
        print("  ❌ HAM10000_metadata.csv introuvable")
        return
    if not os.path.exists(img_dir):
        print("  ❌ ham10000/images/ introuvable")
        return

    df = pd.read_csv(csv_path)
    print(f"  CSV chargé: {len(df)} lignes | colonnes: {list(df.columns)}")

    # Détecter colonnes
    id_col = next((c for c in ['image_id', 'image', 'img_id'] if c in df.columns), df.columns[0])
    dx_col = next((c for c in ['dx', 'diagnosis', 'label']    if c in df.columns), None)

    if not dx_col:
        print(f"  ❌ Colonne dx introuvable. Colonnes: {list(df.columns)}")
        return

    before = len(records)
    for _, row in tqdm(df.iterrows(), total=len(df), desc="  HAM10000"):
        urgence = MAPPING_HAM.get(str(row[dx_col]).strip().lower())
        if not urgence:
            continue
        img_id = str(row[id_col]).strip()
        src    = None
        for ext in ['.jpg', '.jpeg', '.png']:
            c = os.path.join(img_dir, f"{img_id}{ext}")
            if os.path.exists(c):
                src = c
                break
        if src:
            copy_image(src, urgence, 'ham')

    print(f"  ✅ HAM10000: {len(records) - before} images copiées\n")


# ── ISIC 2019 ─────────────────────────────────────────────────────────────────

def relabel_isic():
    csv_path = os.path.join(BASE, "data", "raw", "isic", "ISIC_2019_Training_GroundTruth.csv")
    img_dir  = os.path.join(BASE, "data", "raw", "isic", "images")

    if not os.path.exists(csv_path):
        print("  ❌ ISIC_2019_Training_GroundTruth.csv introuvable")
        return
    if not os.path.exists(img_dir):
        print("  ❌ isic/images/ introuvable")
        return

    df = pd.read_csv(csv_path)
    print(f"  CSV chargé: {len(df)} lignes | colonnes: {list(df.columns)}")

    label_cols = [c for c in ['MEL','NV','BCC','AK','BKL','DF','VASC','SCC']
                  if c in df.columns]
    if not label_cols:
        print(f"  ❌ Colonnes one-hot introuvables. Colonnes: {list(df.columns)}")
        return

    id_col = 'image' if 'image' in df.columns else df.columns[0]
    before = len(records)

    for _, row in tqdm(df.iterrows(), total=len(df), desc="  ISIC 2019"):
        dx = next((c for c in label_cols if str(row.get(c, 0)) == '1' or row.get(c, 0) == 1.0), None)
        if not dx:
            continue
        urgence = MAPPING_ISIC.get(dx)
        if not urgence:
            continue
        img_id = str(row[id_col]).strip()
        src    = None
        for ext in ['.jpg', '.jpeg', '.png']:
            c = os.path.join(img_dir, f"{img_id}{ext}")
            if os.path.exists(c):
                src = c
                break
        if src:
            copy_image(src, urgence, 'isic')

    print(f"  ✅ ISIC 2019: {len(records) - before} images copiées\n")


# ── PH2 ───────────────────────────────────────────────────────────────────────

def relabel_ph2():
    xlsx_path = os.path.join(BASE, "data", "raw", "ph2", "PH2_dataset.xlsx")
    img_dir   = os.path.join(BASE, "data", "raw", "ph2", "images")

    if not os.path.exists(xlsx_path):
        print("  ❌ PH2_dataset.xlsx introuvable")
        return
    if not os.path.exists(img_dir):
        print("  ❌ ph2/images/ introuvable")
        return

    # PH2 excel has weird format — essayer plusieurs skiprows
    df = None
    for skip in [0, 5, 10, 12, 13, 15]:
        try:
            tmp = pd.read_excel(xlsx_path, skiprows=skip, header=0)
            tmp.columns = [str(c).strip() for c in tmp.columns]
            # Chercher colonne qui contient "IMD"
            img_col = next(
                (c for c in tmp.columns
                 if tmp[c].astype(str).str.contains('IMD', na=False).sum() > 5),
                None
            )
            if img_col:
                df = tmp
                print(f"  Excel lu avec skiprows={skip} | img_col='{img_col}'")
                break
        except Exception:
            continue

    if df is None:
        print("  ❌ Impossible de lire PH2_dataset.xlsx")
        return

    # Chercher colonne diagnosis (contient 1, 2, 3)
    diag_col = None
    for c in df.columns:
        vals = set(str(v).strip() for v in df[c].dropna().unique())
        if vals.issubset({'1', '2', '3', '1.0', '2.0', '3.0'}):
            diag_col = c
            break

    if not diag_col:
        # Essayer par nom
        diag_col = next(
            (c for c in df.columns if 'diag' in c.lower() or 'class' in c.lower()),
            None
        )

    if not diag_col:
        print(f"  ❌ Colonne diagnosis introuvable. Colonnes: {list(df.columns)}")
        return

    print(f"  Colonne diagnosis: '{diag_col}'")
    before = len(records)

    for _, row in tqdm(df.iterrows(), total=len(df), desc="  PH2"):
        img_id = str(row[img_col]).strip()
        if not img_id or img_id.lower() == 'nan':
            continue
        try:
            diag = int(float(str(row[diag_col]).strip()))
        except (ValueError, TypeError):
            continue
        urgence = MAPPING_PH2.get(diag)
        if not urgence:
            continue
        src = None
        for ext in ['.bmp', '.jpg', '.jpeg', '.png']:
            c = os.path.join(img_dir, f"{img_id}{ext}")
            if os.path.exists(c):
                src = c
                break
        if src:
            copy_image(src, urgence, 'ph2')

    print(f"  ✅ PH2: {len(records) - before} images copiées\n")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("━━━ Relabeling ━━━\n")
    print("[HAM10000]")
    relabel_ham()
    print("[ISIC 2019]")
    relabel_isic()
    print("[PH2]")
    relabel_ph2()

    if not records:
        print("❌ Aucune image copiée — vérifie les paths et les fichiers CSV/Excel")
    else:
        df_out = pd.DataFrame(records)
        out    = os.path.join(BASE, "data", "processed", "metadata_labeled.csv")
        df_out.to_csv(out, index=False)

        print("━━━ Distribution finale ━━━")
        print(df_out['urgence'].value_counts().sort_index().to_string())
        print(f"\n✅ Total: {len(df_out)} images")
        print(f"⚠️  Non trouvées: {not_found}")
import os
import shutil
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

BASE = r"C:\Users\ismai\OneDrive\Desktop\urgence_system"
SEED = 42

for split in ['train', 'val', 'test']:
    for u in ['P1', 'P2', 'P3', 'P4']:
        os.makedirs(
            os.path.join(BASE, "data", "final", split, "images", u),
            exist_ok=True
        )


def split_images():
    meta = os.path.join(BASE, "data", "processed", "metadata_processed.csv")
    if not os.path.exists(meta):
        print("❌ metadata_processed.csv introuvable — lance 03_preprocess.py")
        return

    df   = pd.read_csv(meta)
    orig = df[~df['augmented']].copy()
    aug  = df[ df['augmented']].copy()

    train, temp = train_test_split(
        orig, test_size=0.30, stratify=orig['urgence'], random_state=SEED
    )
    val, test = train_test_split(
        temp, test_size=0.50, stratify=temp['urgence'], random_state=SEED
    )
    train = pd.concat([train, aug], ignore_index=True)

    for name, subset in [('train', train), ('val', val), ('test', test)]:
        copied = 0
        for _, row in subset.iterrows():
            src = row['image_path']
            if not os.path.exists(src):
                continue
            dst = os.path.join(
                BASE, "data", "final", name,
                "images", row['urgence'],
                os.path.basename(src)
            )
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
                copied += 1

        out = subset.copy()
        out['image_path'] = out.apply(
            lambda r: os.path.join(
                BASE, "data", "final", name,
                "images", r['urgence'],
                os.path.basename(r['image_path'])
            ), axis=1
        )
        out.to_csv(
            os.path.join(BASE, "data", "final", name, "metadata.csv"),
            index=False
        )
        n_orig = len(subset[~subset['augmented']]) if 'augmented' in subset.columns else len(subset)
        print(f"  {name:6s}: {n_orig} originales | {copied} copiées")


def split_text():
    path = os.path.join(BASE, "data", "synthetic", "text_scenarios.csv")
    if not os.path.exists(path):
        print("❌ text_scenarios.csv introuvable — lance 02_generate_text.py")
        return

    df = pd.read_csv(path)
    train, temp = train_test_split(
        df, test_size=0.30, stratify=df['urgence'], random_state=SEED
    )
    val, test = train_test_split(
        temp, test_size=0.50, stratify=temp['urgence'], random_state=SEED
    )
    for name, subset in [('train', train), ('val', val), ('test', test)]:
        subset.to_csv(
            os.path.join(BASE, "data", "final", name, "text.csv"),
            index=False
        )
    print(f"  Texte — train:{len(train)} | val:{len(val)} | test:{len(test)}")


def plot():
    colors = {'P1': '#e74c3c', 'P2': '#e67e22',
               'P3': '#f1c40f', 'P4': '#2ecc71'}
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("Distribution — Urgence Dermatologie", fontsize=13)

    for i, split in enumerate(['train', 'val', 'test']):
        path = os.path.join(BASE, "data", "final", split, "metadata.csv")
        if not os.path.exists(path):
            continue
        df     = pd.read_csv(path)
        counts = df['urgence'].value_counts().sort_index()
        axes[i].bar(counts.index, counts.values,
                    color=[colors[k] for k in counts.index])
        axes[i].set_title(f"{split} ({len(df)})")
        axes[i].set_xlabel("Urgence")
        axes[i].set_ylabel("Images")
        for j, (_, v) in enumerate(counts.items()):
            axes[i].text(j, v + 5, str(v), ha='center', fontsize=9)

    plt.tight_layout()
    out = os.path.join(BASE, "data", "final", "distribution.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"\n  Graphique -> {out}")


if __name__ == "__main__":
    print("━━━ Split Images ━━━")
    split_images()
    print("\n━━━ Split Texte ━━━")
    split_text()
    plot()

    print("\n" + "=" * 45)
    print("✅ PHASE 1 TERMINEE — PRET POUR TRAINING")
    print("=" * 45)

    for split in ['train', 'val', 'test']:
        path = os.path.join(BASE, "data", "final", split, "metadata.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            print(f"\n  [{split}]")
            print(df['urgence'].value_counts().sort_index().to_string())
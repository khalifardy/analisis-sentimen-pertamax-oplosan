"""
Labeling Module
===============
Fungsi untuk:
- Load dan cleaning data dari 2 annotator
- Menghitung Inter-Annotator Agreement (Cohen's Kappa)
- Menampilkan distribusi label per dataset
"""

import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score


# Mapping label
LABEL_NAMES_MAP = {0: 'Negatif', 1: 'Netral', 2: 'Positif'}
LABEL_TEXT_TO_NUM = {'negatif': 0, 'netral': 1, 'positif': 2}


def load_and_clean_data(path_annotator1='data/raw/data_annotator1.csv',
                         path_annotator2='data/raw/data_annotator2_fadly.csv'):
    """
    Load dan bersihkan data dari kedua annotator.

    Returns:
    - df_a: DataFrame annotator 1 (kolom: clean_text, label)
    - df_b: DataFrame annotator 2 (kolom: clean_text, label)
    """
    # ========== DATASET A (Annotator 1) ==========
    df_a = pd.read_csv(path_annotator1)
    df_a = df_a[['clean_text', 'Label']].rename(columns={'Label': 'label'})

    # Hanya ambil label valid (0, 1, 2)
    df_a['label'] = pd.to_numeric(df_a['label'], errors='coerce')
    df_a = df_a[df_a['label'].isin([0, 1, 2])].copy()
    df_a['label'] = df_a['label'].astype(int)

    # Buang teks kosong/NaN
    df_a = df_a[df_a['clean_text'].notna()]
    df_a = df_a[df_a['clean_text'].str.strip() != '']
    df_a = df_a.reset_index(drop=True)

    print(f"Dataset A (Annotator 1) - Total: {len(df_a)}")
    print(df_a['label'].value_counts().sort_index())

    # ========== DATASET B (Annotator 2 - Fadly) ==========
    df_b = pd.read_csv(path_annotator2)
    df_b = df_b[['clean_text', 'label']].copy()

    # Konversi label text ke numerik
    df_b['label'] = df_b['label'].apply(
        lambda x: LABEL_TEXT_TO_NUM.get(str(x).strip().lower(), x)
    )
    df_b['label'] = pd.to_numeric(df_b['label'], errors='coerce')
    df_b = df_b[df_b['label'].isin([0, 1, 2])].copy()
    df_b['label'] = df_b['label'].astype(int)

    # Buang teks kosong/NaN
    df_b = df_b[df_b['clean_text'].notna()]
    df_b = df_b[df_b['clean_text'].str.strip() != '']
    df_b = df_b.reset_index(drop=True)

    print(f"\nDataset B (Annotator 2 - Fadly) - Total: {len(df_b)}")
    print(df_b['label'].value_counts().sort_index())

    return df_a, df_b


def compute_inter_annotator_agreement(df_a, df_b):
    """
    Hitung Inter-Annotator Agreement antara 2 annotator.

    Parameters:
    - df_a: DataFrame annotator 1 (kolom: clean_text, label)
    - df_b: DataFrame annotator 2 (kolom: clean_text, label)

    Returns:
    - dict berisi agreement_rate, cohen_kappa, jumlah data
    """
    df_merged = pd.merge(
        df_a[['clean_text', 'label']].rename(columns={'label': 'label_a'}),
        df_b[['clean_text', 'label']].rename(columns={'label': 'label_b'}),
        on='clean_text', how='inner'
    )

    agreement_rate = (df_merged['label_a'] == df_merged['label_b']).mean()
    kappa = cohen_kappa_score(df_merged['label_a'], df_merged['label_b'])

    print(f"\n{'='*50}")
    print(f"INTER-ANNOTATOR AGREEMENT")
    print(f"{'='*50}")
    print(f"Jumlah data yang bisa dibandingkan: {len(df_merged)}")
    print(f"Agreement Rate : {agreement_rate:.4f} ({agreement_rate*100:.2f}%)")
    print(f"Cohen's Kappa  : {kappa:.4f}")

    # Interpretasi Kappa
    if kappa < 0.20:
        interp = "Slight agreement"
    elif kappa < 0.40:
        interp = "Fair agreement"
    elif kappa < 0.60:
        interp = "Moderate agreement"
    elif kappa < 0.80:
        interp = "Substantial agreement"
    else:
        interp = "Almost perfect agreement"
    print(f"Interpretasi   : {interp}")

    return {
        'n_comparable': len(df_merged),
        'agreement_rate': agreement_rate,
        'cohen_kappa': kappa,
        'interpretation': interp
    }


def show_label_distribution(df, dataset_name):
    """
    Tampilkan tabel distribusi label (untuk Bab 4, Tabel 4.4.5).

    Parameters:
    - df: DataFrame dengan kolom 'label'
    - dataset_name: nama dataset untuk judul tabel
    """
    total = len(df)
    dist = df['label'].value_counts().sort_index()

    print(f"\n{'='*45}")
    print(f"DISTRIBUSI KATEGORI - {dataset_name}")
    print(f"{'='*45}")
    print(f"{'Kategori':<12} {'Jumlah':>8} {'Persentase':>12}")
    print(f"{'-'*35}")

    for label_val in [0, 1, 2]:
        count = dist.get(label_val, 0)
        pct = (count / total) * 100
        print(f"{LABEL_NAMES_MAP[label_val]:<12} {count:>8} {pct:>10.2f}%")

    print(f"{'-'*35}")
    print(f"{'Total':<12} {total:>8} {'100.00%':>12}")

    return dist


def show_balance_distribution(y_data, method_name):
    """
    Tampilkan distribusi setelah balancing.

    Parameters:
    - y_data: array 1D label (bukan one-hot)
    - method_name: nama metode balancing
    """
    unique, counts = np.unique(y_data, return_counts=True)
    total = counts.sum()

    print(f"\n--- Distribusi {method_name} ---")
    for u, c in zip(unique, counts):
        print(f"  {LABEL_NAMES_MAP[int(u)]}: {c} ({c/total*100:.2f}%)")
    print(f"  Total: {total}")

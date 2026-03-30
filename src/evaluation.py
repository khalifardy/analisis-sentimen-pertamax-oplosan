"""
Evaluation Module
=================
Fungsi untuk:
- Tabel rangkuman perbandingan hasil
- Confusion matrix (normal + normalized)
- ROC curve
- Learning curve
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc
)
from sklearn.preprocessing import label_binarize

from src.models import MODEL_CONFIGS
from src.balancing import BALANCING_METHODS


# ============= LABEL NAMES =============
LABEL_NAMES = ['Negatif', 'Netral', 'Positif']


# ============= TABEL RANGKUMAN =============

def print_comparison_table(all_results, metric='accuracy'):
    """
    Cetak tabel perbandingan kedua dataset.

    Parameters:
    - all_results: dict dari run_all_experiments()
    - metric: 'accuracy', 'precision', 'recall', atau 'f1'
    """
    bal_names = list(BALANCING_METHODS.keys())
    model_names = list(MODEL_CONFIGS.keys())

    for ds_name in ['Dataset_A', 'Dataset_B']:
        if ds_name not in all_results:
            continue

        print(f"\n{'=' * 95}")
        print(f"RATA-RATA {metric.upper()} (3 RUNS) - {ds_name}")
        print(f"{'=' * 95}")

        header = f"{'Model':<22}" + "".join(f"{b:>16}" for b in bal_names)
        print(header)
        print("-" * len(header))

        for model_name in model_names:
            row = f"{model_name:<22}"
            for bal_name in bal_names:
                r = all_results[ds_name][bal_name][model_name][metric]
                row += f"  {r['mean']:.4f}±{r['std']:.4f}"
            print(row)

    # Tabel selisih
    if 'Dataset_A' in all_results and 'Dataset_B' in all_results:
        print(f"\n{'=' * 95}")
        print(f"SELISIH {metric.upper()}: Dataset A - Dataset B")
        print(f"{'=' * 95}")

        header = f"{'Model':<22}" + "".join(f"{b:>16}" for b in bal_names)
        print(header)
        print("-" * len(header))

        for model_name in model_names:
            row = f"{model_name:<22}"
            for bal_name in bal_names:
                a = all_results['Dataset_A'][bal_name][model_name][metric]['mean']
                b = all_results['Dataset_B'][bal_name][model_name][metric]['mean']
                diff = a - b
                sign = "+" if diff >= 0 else ""
                row += f"       {sign}{diff:.4f}  "
            print(row)


def print_all_metrics_table(all_results):
    """Cetak tabel untuk semua metrik."""
    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        print_comparison_table(all_results, metric)


# ============= CONFUSION MATRIX =============

def plot_confusion_matrix(y_true, y_pred, title, save_path=None):
    """
    Plot confusion matrix (normal + normalized).

    Parameters:
    - y_true: array label sebenarnya
    - y_pred: array label prediksi
    - title: judul plot
    - save_path: path untuk simpan gambar (opsional)
    """
    cm = confusion_matrix(y_true, y_pred)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Normal
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES, ax=axes[0])
    axes[0].set_title(f'{title} - Count')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')

    # Normalized
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES, ax=axes[1])
    axes[1].set_title(f'{title} - Normalized')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Actual')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")

    plt.show()


# ============= ROC CURVE =============

def plot_roc_curve(y_true, y_pred_prob, title, save_path=None):
    """
    Plot ROC curve per kelas.

    Parameters:
    - y_true: array label sebenarnya (1D, bukan one-hot)
    - y_pred_prob: array probabilitas prediksi (n_samples, 3)
    - title: judul plot
    - save_path: path untuk simpan gambar (opsional)
    """
    y_bin = label_binarize(y_true, classes=[0, 1, 2])

    fig, ax = plt.subplots(figsize=(8, 6))

    colors = ['#e74c3c', '#3498db', '#2ecc71']

    for i, (name, color) in enumerate(zip(LABEL_NAMES, colors)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_pred_prob[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f'{name} (AUC = {roc_auc:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curve - {title}')
    ax.legend(loc='lower right')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")

    plt.show()


# ============= LEARNING CURVE =============

def plot_learning_curve(history, title, save_path=None):
    """
    Plot learning curve (loss + accuracy) dari training history.

    Parameters:
    - history: Keras History object
    - title: judul plot
    - save_path: path untuk simpan gambar (opsional)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    axes[0].plot(history.history['loss'], label='Train Loss', color='#e74c3c')
    axes[0].plot(history.history['val_loss'], label='Val Loss', color='#3498db')
    axes[0].set_title(f'{title} - Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(history.history['accuracy'], label='Train Accuracy', color='#e74c3c')
    axes[1].plot(history.history['val_accuracy'], label='Val Accuracy', color='#3498db')
    axes[1].set_title(f'{title} - Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")

    plt.show()


# ============= CLASSIFICATION REPORT =============

def print_classification_report(y_true, y_pred, title=''):
    """Print classification report dengan label names."""
    if title:
        print(f"\n--- Classification Report: {title} ---")
    print(classification_report(y_true, y_pred, target_names=LABEL_NAMES))


# ============= BEST MODEL FINDER =============

def find_best_model(all_results, metric='accuracy'):
    """
    Cari kombinasi terbaik per dataset.

    Returns:
    - dict per dataset dengan info model terbaik
    """
    best = {}

    for ds_name in all_results:
        best_score = -1
        best_info = {}

        for bal_name in all_results[ds_name]:
            for model_name in all_results[ds_name][bal_name]:
                score = all_results[ds_name][bal_name][model_name][metric]['mean']
                if score > best_score:
                    best_score = score
                    best_info = {
                        'dataset': ds_name,
                        'balancing': bal_name,
                        'model': model_name,
                        'score_mean': score,
                        'score_std': all_results[ds_name][bal_name][model_name][metric]['std']
                    }

        best[ds_name] = best_info
        print(f"\n{ds_name} - Best {metric}:")
        print(f"  Model    : {best_info['model']}")
        print(f"  Balancing: {best_info['balancing']}")
        print(f"  Score    : {best_info['score_mean']:.4f} ± {best_info['score_std']:.4f}")

    return best

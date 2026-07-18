"""
Baseline No-Balancing Experiment
=================================
Menjalankan 7 model klasifikasi TANPA balancing pada dataset KONSENSUS
(33,351 tweet dari dataset_saya + dataset_fadly, Kappa 0.8604).

Pipeline:
  1. Merge dataset_saya.csv + dataset_fadly.csv via full_text
  2. Filter agreement (label sama) -> 33,351 rows
  3. Preprocessing (case folding, cleansing, slang, stopword, stemming)
  4. TF-IDF vectorization
  5. Train 7 model x 3 seeds (tanpa balancing)

Output: results/baseline_no_balancing.json
"""

import os
import sys
import json
import warnings
import numpy as np
import gc
import pandas as pd

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras import backend as K

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import keras_tuner as kt

from src.preprocessing import full_preprocessing
from src.models import (
    create_mlp_baseline, create_mlp_advance, make_tuner_builder,
    create_naive_bayes, create_svm, create_random_forest,
    create_logistic_regression, get_callbacks, MODEL_CONFIGS
)

NUM_RUNS = 3
SEEDS = [42, 123, 456]


def run_single_baseline(X_train, y_train_1d, X_test, y_test_1d,
                        model_type, seed, run_idx=0):
    """Train & evaluate 1 model tanpa balancing."""
    callbacks = get_callbacks()
    y_train_oh = tf.keras.utils.to_categorical(y_train_1d, num_classes=3)
    y_test_oh = tf.keras.utils.to_categorical(y_test_1d, num_classes=3)

    if model_type == 'mlp_baseline':
        model = create_mlp_baseline(X_train.shape[1])
        model.fit(X_train, y_train_oh, epochs=50, batch_size=32,
                  validation_split=0.15, verbose=0)
        y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
        del model; gc.collect(); K.clear_session()

    elif model_type == 'mlp_advance':
        model = create_mlp_advance(X_train.shape[1])
        model.fit(X_train, y_train_oh, epochs=50, batch_size=32,
                  validation_split=0.15, callbacks=callbacks, verbose=0)
        y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
        del model; gc.collect(); K.clear_session()

    elif model_type == 'mlp_tuner':
        builder = make_tuner_builder(X_train.shape[1])
        tuner = kt.RandomSearch(
            builder, objective='val_accuracy', max_trials=5,
            executions_per_trial=1,
            directory=f'tuner_baseline_run{run_idx}',
            project_name='sentiment',
            overwrite=True
        )
        tuner.search(X_train, y_train_oh, epochs=50, batch_size=32,
                     validation_split=0.15, callbacks=callbacks, verbose=0)
        best_model = tuner.get_best_models(1)[0]
        y_pred = np.argmax(best_model.predict(X_test, verbose=0), axis=1)
        del best_model, tuner; gc.collect(); K.clear_session()

    elif model_type == 'naive_bayes':
        model = create_naive_bayes()
        model.fit(X_train, y_train_1d)
        y_pred = model.predict(X_test)

    elif model_type == 'svm':
        model = create_svm(random_state=seed)
        model.fit(X_train, y_train_1d)
        y_pred = model.predict(X_test)

    elif model_type == 'random_forest':
        model = create_random_forest(random_state=seed)
        model.fit(X_train, y_train_1d)
        y_pred = model.predict(X_test)

    elif model_type == 'logistic_regression':
        model = create_logistic_regression(random_state=seed)
        model.fit(X_train, y_train_1d)
        y_pred = model.predict(X_test)

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # For non-MLP models, y_test_1d is already 1D
    if model_type in ('mlp_baseline', 'mlp_advance', 'mlp_tuner'):
        y_true = np.argmax(y_test_oh, axis=1)
    else:
        y_true = y_test_1d

    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, average='weighted')),
        'recall': float(recall_score(y_true, y_pred, average='weighted')),
        'f1': float(f1_score(y_true, y_pred, average='weighted')),
    }


def run_baseline_multi(X_train, y_train_1d, X_test, y_test_1d,
                       model_type, n_runs=NUM_RUNS, seeds=SEEDS):
    """Jalankan 1 model sebanyak n_runs dengan seed berbeda."""
    metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}

    for run_idx, seed in enumerate(seeds[:n_runs]):
        print(f"  Run {run_idx + 1}/{n_runs} (seed={seed})...", end=' ', flush=True)

        result = run_single_baseline(
            X_train, y_train_1d, X_test, y_test_1d,
            model_type, seed, run_idx=run_idx
        )

        for k, v in result.items():
            metrics[k].append(v)

        print(f"Acc={result['accuracy']:.4f} | F1={result['f1']:.4f}")

    summary = {}
    for k, v in metrics.items():
        summary[k] = {
            'mean': np.mean(v),
            'std': np.std(v),
            'runs': v
        }

    print(f"  >> RATA-RATA: Acc={summary['accuracy']['mean']:.4f} "
          f"(+/-{summary['accuracy']['std']:.4f}) | "
          f"F1={summary['f1']['mean']:.4f} "
          f"(+/-{summary['f1']['std']:.4f})")

    return summary


def convert_results(obj):
    if isinstance(obj, dict):
        return {k: convert_results(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_results(i) for i in obj]
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


def main():
    print("=" * 80)
    print("BASELINE NO-BALANCING EXPERIMENT")
    print("Dataset: Consensus dari dataset_saya + dataset_fadly (Kappa 0.8604)")
    print("7 Models x 3 Seeds = 21 runs")
    print("=" * 80)

    # ========== 1. LOAD & MERGE ==========
    print("\n[STEP 1] Loading & merging annotator datasets...")
    df_saya = pd.read_csv('data/processed/dataset_saya.csv')
    df_fadly = pd.read_csv('data/processed/dataset_fadly.csv')
    print(f"  dataset_saya : {len(df_saya)} rows")
    print(f"  dataset_fadly: {len(df_fadly)} rows")

    df_merged = pd.merge(
        df_saya[['full_text', 'label']].rename(columns={'label': 'label_saya'}),
        df_fadly[['full_text', 'label']].rename(columns={'label': 'label_fadly'}),
        on='full_text', how='inner'
    )
    print(f"  Merged (inner): {len(df_merged)} rows")

    # Filter agreement
    df_agree = df_merged[df_merged['label_saya'] == df_merged['label_fadly']].copy()
    df_agree['label'] = df_agree['label_saya'].astype(int)
    df_agree = df_agree[['full_text', 'label']]
    print(f"  Consensus (agree): {len(df_agree)} rows")
    print(f"  Label distribution:\n{df_agree['label'].value_counts()}")

    # ========== 2. PREPROCESSING ==========
    print("\n[STEP 2] Text preprocessing (5-15 menit)...")
    df_clean = full_preprocessing(df_agree, kolom='full_text', use_stemming=True)
    print(f"  After preprocessing: {len(df_clean)} rows")

    # ========== 3. TF-IDF + SPLIT ==========
    print("\n[STEP 3] TF-IDF vectorization & train/test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        df_clean['clean_text'], df_clean['label'],
        test_size=0.2, random_state=SEEDS[0], stratify=df_clean['label']
    )

    tfidf = TfidfVectorizer(
        max_features=5000, min_df=2, max_df=0.95, ngram_range=(1, 2)
    )
    X_train_tf = tfidf.fit_transform(X_train)
    X_test_tf = tfidf.transform(X_test)

    print(f"  Train: {X_train_tf.shape}")
    print(f"  Test : {X_test_tf.shape}")
    print(f"  y_train distribution:\n{pd.Series(y_train).value_counts()}")

    # ========== 4. EXPERIMENT ==========
    print("\n[STEP 4] Running 7 models x 3 seeds...")
    print("=" * 80)

    results = {}
    for model_name, model_type in MODEL_CONFIGS.items():
        print(f"\n{'='*80}")
        print(f">> {model_name}")
        print(f"{'='*80}")
        results[model_name] = run_baseline_multi(
            X_train_tf, y_train, X_test_tf, y_test,
            model_type, n_runs=NUM_RUNS, seeds=SEEDS
        )

    # ========== 5. SAVE ==========
    all_results = {'Dataset_Konsensus': {'No_Balancing': results}}

    os.makedirs('results', exist_ok=True)
    output_path = 'results/baseline_no_balancing.json'
    with open(output_path, 'w') as f:
        json.dump(convert_results(all_results), f, indent=2)
    print(f"\n\n[OK] Hasil disimpan ke: {output_path}")

    # ========== 6. PRINT TABLE ==========
    print("\n" + "=" * 80)
    print("BASELINE NO-BALANCING - DATASET KONSENSUS")
    print("=" * 80)
    print(f"{'Model':<22} {'F1 (%)':>12} {'Acc (%)':>12}")
    print("-" * 50)
    for model_name in MODEL_CONFIGS:
        r = results[model_name]
        print(f"{model_name:<22} {r['f1']['mean']*100:>8.2f} +/-{r['f1']['std']*100:.2f}  {r['accuracy']['mean']*100:>7.2f} +/-{r['accuracy']['std']*100:.2f}")

    print("\n" + "=" * 80)
    print("SELESAI!")
    print("=" * 80)


if __name__ == "__main__":
    main()

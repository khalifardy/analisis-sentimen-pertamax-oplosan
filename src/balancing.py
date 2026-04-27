"""
Balancing & Experiment Runner Module
=====================================
Fungsi untuk:
- Menjalankan balancing data (5 metode)
- Menjalankan eksperimen 3x run per kombinasi model+balancing
- Menjalankan seluruh eksperimen untuk kedua dataset
"""

import numpy as np
import gc

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN, SMOTETomek

import keras_tuner as kt

from src.models import (
    create_mlp_baseline, create_mlp_advance, make_tuner_builder,
    create_naive_bayes, create_svm, create_random_forest,
    create_logistic_regression, get_callbacks, MODEL_CONFIGS
)


# ============= KONFIGURASI =============

NUM_RUNS = 3
SEEDS = [42, 123, 456]

BALANCING_METHODS = {
    'ROS':        RandomOverSampler,
    'SMOTE':      SMOTE,
    'RUS':        RandomUnderSampler,
    'SMOTEENN':   SMOTEENN,
    'SMOTETomek': SMOTETomek,
}


# ============= FEATURE EXTRACTION =============

def prepare_features(df, dataset_name='',kolom_x='clean_text',kolom_y='label',seed=42):
    """
    TF-IDF Vectorization + one-hot encoding.

    Parameters:
    - df: DataFrame dengan kolom 'clean_text' dan 'label'
    - dataset_name: nama dataset untuk print

    Returns:
    - X: sparse matrix TF-IDF
    - y: array 1D label
    - y_one_hot: array one-hot encoded label
    - tfidf: fitted TfidfVectorizer
    """
    
    X = df[kolom_x]
    y = df[kolom_y]
    
     
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )
    
    tfidf = TfidfVectorizer(
        max_features=5000,
        min_df=2,
        max_df=0.95,
        ngram_range=(1, 2)
    )

    X_train_tf = tfidf.fit_transform(X_train)
    X_test_tf = tfidf.transform(X_test)
    
    y_one_hot_train = to_categorical(y_train, num_classes=3)
    y_one_hot_test = to_categorical(y_test, num_classes=3)

    if dataset_name:
        print(f"\n{dataset_name}:")
        print(f"  Shape fitur TF-IDF Train: {X_train_tf.shape}")
        print(f"  Shape fitur TF-IDF Test: {X_test_tf.shape}")
        print(f"  Jumlah fitur train: {X_train_tf.shape[1]}")
        print(f"  Jumlah fitur test: {X_test_tf.shape[1]}")

    return X_train_tf,y_one_hot_train, X_test_tf,y_one_hot_test,tfidf

# ============= SINGLE RUN EXPERIMENT =============

def run_single_experiment(X_train, y_one_hot_train,X_test, y_one_hot_test, model_type, bal_class, seed,
                          run_idx=0, bal_name=''):
    """
    Jalankan 1 kali eksperimen (1 run).

    Returns:
    - dict metrik: accuracy, precision, recall, f1
    """
    callbacks = get_callbacks()
    
    # 1. Split
    #X_train, X_test, y_train, y_test = train_test_split(
        #X, y_one_hot, test_size=0.2, random_state=seed
    #)

    # 2. Balancing
    sampler = bal_class(random_state=seed)
    X_res, y_res = sampler.fit_resample(X_train, y_one_hot_train)

   

    # 3. Train & Predict
    if model_type == 'mlp_baseline':
        model = create_mlp_baseline(X_res.shape[1])
        model.fit(X_res, y_res, epochs=50, batch_size=32,
                  validation_split=0.15, verbose=0)
        y_pred = np.argmax(model.predict(X_test), axis=1)
        y_true = np.argmax(y_one_hot_test, axis=1)
        del model; gc.collect(); K.clear_session()

    elif model_type == 'mlp_advance':
        model = create_mlp_advance(X_res.shape[1])
        model.fit(X_res, y_res, epochs=50, batch_size=32,
                  validation_split=0.15, callbacks=callbacks, verbose=0)
        y_pred = np.argmax(model.predict(X_test), axis=1)
        y_true = np.argmax(y_one_hot_test, axis=1)
        del model; gc.collect(); K.clear_session()

    elif model_type == 'mlp_tuner':
        builder = make_tuner_builder(X_res.shape[1])
        tuner = kt.RandomSearch(
            builder, objective='val_accuracy', max_trials=5,
            executions_per_trial=1,
            directory=f'tuner_{bal_name}_run{run_idx}',
            project_name='sentiment',
            overwrite=True
        )
        tuner.search(X_res, y_res, epochs=50, batch_size=32,
                     validation_split=0.15, callbacks=callbacks, verbose=0)
        best_model = tuner.get_best_models(1)[0]
        y_pred = np.argmax(best_model.predict(X_test), axis=1)
        y_true = np.argmax(y_one_hot_test, axis=1)
        del best_model, tuner; gc.collect(); K.clear_session()

    elif model_type == 'naive_bayes':
        y_train_1d = np.argmax(y_res, axis=1)
        y_true = np.argmax(y_one_hot_test, axis=1)
        model = create_naive_bayes()
        model.fit(X_res, y_train_1d)
        y_pred = model.predict(X_test)

    elif model_type == 'svm':
        y_train_1d = np.argmax(y_res, axis=1)
        y_true = np.argmax(y_one_hot_test, axis=1)
        model = create_svm(random_state=seed)
        model.fit(X_res, y_train_1d)
        y_pred = model.predict(X_test)

    elif model_type == 'random_forest':
        y_train_1d = np.argmax(y_res, axis=1)
        y_true = np.argmax(y_one_hot_test, axis=1)
        model = create_random_forest(random_state=seed)
        model.fit(X_res, y_train_1d)
        y_pred = model.predict(X_test)

    elif model_type == 'logistic_regression':
        y_train_1d = np.argmax(y_res, axis=1)
        y_true = np.argmax(y_one_hot_test, axis=1)
        model = create_logistic_regression(random_state=seed)
        model.fit(X_res, y_train_1d)
        y_pred = model.predict(X_test)

    else:
        raise ValueError(f"model_type tidak dikenal: {model_type}")

    # 4. Hitung metrik
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1': f1_score(y_true, y_pred, average='weighted'),
    }


# ============= MULTI-RUN EXPERIMENT =============

def run_multi_experiment(X_train, y_one_hot_train,X_test, y_one_hot_test, model_type, bal_name, bal_class,
                          n_runs=NUM_RUNS, seeds=SEEDS):
    """
    Jalankan 1 kombinasi (model + balancing) sebanyak n_runs.

    Returns:
    - dict metrik dengan mean, std, dan detail per run
    """
    metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}

    for run_idx, seed in enumerate(seeds[:n_runs]):
        print(f"  Run {run_idx + 1}/{n_runs} (seed={seed})...", end=' ')

        result = run_single_experiment(
            X_train, y_one_hot_train,X_test, y_one_hot_test, model_type, bal_class, seed,
            run_idx=run_idx, bal_name=bal_name
        )

        for k, v in result.items():
            metrics[k].append(v)

        print(f"Acc={result['accuracy']:.4f} | F1={result['f1']:.4f}")

    # Rangkuman
    summary = {}
    for k, v in metrics.items():
        summary[k] = {
            'mean': np.mean(v),
            'std': np.std(v),
            'runs': v
        }

    print(f"  >> RATA-RATA: Acc={summary['accuracy']['mean']:.4f} "
          f"(±{summary['accuracy']['std']:.4f}) | "
          f"F1={summary['f1']['mean']:.4f} "
          f"(±{summary['f1']['std']:.4f})")

    return summary


# ============= RUN ALL EXPERIMENTS =============

def run_all_experiments(df_a, df_b, n_runs=NUM_RUNS, seeds=SEEDS, kolom_x_a='clean_text',kolom_x_b='clean_text',kolom_y_a='label',kolom_y_b='label'):
    """
    Jalankan semua eksperimen untuk kedua dataset.

    Parameters:
    - df_a: DataFrame dataset annotator 1
    - df_b: DataFrame dataset annotator 2
    - n_runs: jumlah run per kombinasi
    - seeds: list random seed

    Returns:
    - all_results: nested dict [dataset][balancing][model] = metrik
    """
    # Prepare features
    print("\n" + "=" * 60)
    print("PREPARE FEATURES")
    print("=" * 60)
    X_train_tf_a,y_one_hot_train_a, X_test_tf_a,y_one_hot_test_a,tfidf_a = prepare_features(df_a, "Dataset A (Annotator 1)",kolom_x=kolom_x_a,kolom_y=kolom_y_a,seed=seeds[0])
    X_train_tf_b,y_one_hot_train_b, X_test_tf_b,y_one_hot_test_b,tfidf_b = prepare_features(df_b, "Dataset B (Annotator 2)",kolom_x=kolom_x_b,kolom_y=kolom_y_b,seed=seeds[0])

    datasets = {
        'Dataset_A': (X_train_tf_a,y_one_hot_train_a, X_test_tf_a,y_one_hot_test_a),
        'Dataset_B': (X_train_tf_b,y_one_hot_train_b, X_test_tf_b,y_one_hot_test_b),
    }

    all_results = {}

    for ds_name, (X_train_ds, y_train_ds, X_test_ds, y_test_ds) in datasets.items():
        all_results[ds_name] = {}

        print(f"\n{'#' * 70}")
        print(f"  DATASET: {ds_name}")
        print(f"{'#' * 70}")

        for bal_name, bal_class in BALANCING_METHODS.items():
            all_results[ds_name][bal_name] = {}

            for model_name, model_type in MODEL_CONFIGS.items():
                print(f"\n>> {ds_name} | {bal_name} | {model_name}")
                all_results[ds_name][bal_name][model_name] = run_multi_experiment(
                    X_train_ds, y_train_ds, X_test_ds, y_test_ds, model_type, bal_name, bal_class,
                    n_runs=n_runs, seeds=seeds
                )

    return all_results

def run_experiments_single_dataset(df, dataset_name='Dataset_kesepakatan',
                                   kolom_x='clean_text', kolom_y='label',
                                   n_runs=NUM_RUNS, seeds=SEEDS):
    """
    Jalankan semua eksperimen untuk 1 dataset (kesepakatan)
    
    parameters:
    - df: DataFrame dataset kesepakatan
    - dataset_name: nama dataset
    - kolom_x: nama kolom teks
    - kolom_y: nama kolom label
    - n_runs: jumlah run per kombinasi
    - seeds: list random seed
    
    Returns:
    - all_results: nested dict [dataset][balancing][model] = metrik
    """
    
    print("\n" + "=" * 60)
    print("PREPARE FEATURES")
    print("="*60)
    
    X_train, y_train_oh, X_test, y_test_oh, tfidf = prepare_features(
        df, dataset_name,kolom_x=kolom_x,kolom_y=kolom_y,seed=seeds[0]
        )
    
    all_results = {dataset_name:{}}
    
    print(f"\n{'#'*70}")
    print(f" DATASET: {dataset_name}")
    print(f"{'#'*70}")
    
    for bal_name, bal_class in BALANCING_METHODS.items():
        all_results[dataset_name][bal_name] = {}
        
        for model_name, model_type in MODEL_CONFIGS.items():
            print(f"\n>> {dataset_name} | {bal_name} | {model_name}")
            all_results[dataset_name][bal_name][model_name] = run_multi_experiment(
                X_train, y_train_oh, X_test, y_test_oh, 
                model_type, bal_name, bal_class,
                n_runs=n_runs, seeds=seeds
                )
    
    return all_results
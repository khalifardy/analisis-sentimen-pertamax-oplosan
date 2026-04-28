import marimo

__generated_with = "0.22.0"
app = marimo.App(width="medium", auto_download=["ipynb"])


@app.cell
def _():
    import pandas as pd
    from src.balancing import run_experiments_single_dataset #run_all_experiments

    return pd, run_experiments_single_dataset


@app.cell
def _(pd):
    #df_rendika = pd.read_csv('data/processed/preprocessed_dataset_rendika.csv')
    #df_fadly = pd.read_csv('data/processed/preprocessed_dataset_fadly.csv')

    df_kesepakatan = pd.read_csv('data/processed/preprocessed_dataset_kesepakatan.csv')
    df_kesepakatan
    return (df_kesepakatan,)


@app.cell
def _():
    import os
    os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

    import tensorflow as tf

    gpus = tf.config.list_physical_device('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        
    return (os,)


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #Eksperimen
    """)
    return


@app.cell
def _(df_kesepakatan, run_experiments_single_dataset):
    all_results = run_experiments_single_dataset(df_kesepakatan, kolom_y='label')
    return (all_results,)


@app.cell
def _():
    import json
    import numpy as np

    return json, np


@app.cell
def _(all_results, json, np):
    def convert_results(obj):
        if isinstance(obj, dict):
            return {k: convert_results(v) for k,v in obj.items()}
        elif isinstance(obj, list):
            return [convert_results(i) for i in obj]
        elif isinstance(obj,(np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return obj


    with open('results/all_results.json)','w') as f:
        json.dump(convert_results(all_results), f,indent=2)

    #untuk load json
    #with open('results/all_results.json', 'r') as f:
        #all_results =json.load(f)
    return


@app.cell
def _():
    from src.evaluation import print_all_metrics_table, find_best_model

    return find_best_model, print_all_metrics_table


@app.cell
def _(all_results, find_best_model, print_all_metrics_table):
    print_all_metrics_table(all_results)
    best_acc = find_best_model(all_results, metric='accuracy')
    best_pre = find_best_model(all_results, metric='precision')
    best_reca = find_best_model(all_results, metric='recall')
    best_f1 = find_best_model(all_results, metric='f1')
    return


@app.cell
def _():
    # EValuasi

    import os, gc
    import numpy as np
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, f1_score
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras import backend as K
    import keras_tuner as kt

    from imblearn.over_sampling import RandomOverSampler, SMOTE
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.combine import SMOTEENN,SMOTETomek

    from src.models import (
        create_mlp_baseline, create_mlp_advance, make_tuner_builder,create_naive_bayes, create_logistic_regression, create_random_forest, create_svm, get_callbacks, MODEL_CONFIGS
    )

    from src.balancing import BALANCING_METHODS
    from src.evaluation import (
        plot_confusion_matrix,
        plot_roc_curve,
        plot_learning_curve,
        print_all_metrics_table,
        print_classification_report,
        find_best_model
    )

    SEED=42
    return (
        BALANCING_METHODS,
        K,
        MODEL_CONFIGS,
        SEED,
        TfidfVectorizer,
        accuracy_score,
        create_logistic_regression,
        create_mlp_advance,
        create_mlp_baseline,
        create_naive_bayes,
        create_random_forest,
        create_svm,
        f1_score,
        find_best_model,
        gc,
        get_callbacks,
        kt,
        make_tuner_builder,
        np,
        os,
        pd,
        plot_confusion_matrix,
        plot_learning_curve,
        plot_roc_curve,
        print_all_metrics_table,
        print_classification_report,
        to_categorical,
        train_test_split,
    )


@app.cell
def _(SEED, TfidfVectorizer, df_kesepakatan, to_categorical, train_test_split):
    def prepare(df, kx, ky):
        X_tr, X_te, y_tr, y_te = train_test_split(df[kx], df[ky], test_size=0.2, random_state=SEED)

        tfidf = TfidfVectorizer(max_features=5000, min_df=2, max_df=0.95, ngram_range=(1,2))

        return tfidf.fit_transform(X_tr), tfidf.transform(X_te), to_categorical(y_tr,3), to_categorical(y_te,3)

    data_prepared = {
        'Dataset_A': prepare(df_kesepakatan, 'clean_text', 'Label'),
        #'Dataset_B': prepare(df_fadly, 'clean_text', 'label'),
    }
    return (data_prepared,)


@app.cell
def _(
    BALANCING_METHODS,
    K,
    MODEL_CONFIGS,
    SEED,
    accuracy_score,
    create_logistic_regression,
    create_mlp_advance,
    create_mlp_baseline,
    create_naive_bayes,
    create_random_forest,
    create_svm,
    data_prepared,
    f1_score,
    gc,
    get_callbacks,
    kt,
    make_tuner_builder,
    np,
    os,
    plot_confusion_matrix,
    plot_learning_curve,
    plot_roc_curve,
    print_classification_report,
):
    DATASET_FOLDERS = {'Dataset_A': 'dataset_kesepakatan'}
    total = len(data_prepared) * len(BALANCING_METHODS) * len(MODEL_CONFIGS)
    counter = 0

    for ds_name, (X_train, X_test, y_train_oh, y_test_oh) in data_prepared.items():
        y_true = np.argmax(y_test_oh, axis=1)

        for model_name, model_type in MODEL_CONFIGS.items():
            save_dir = f'results/{DATASET_FOLDERS[ds_name]}/{model_name.replace(" ","_")}'
            os.makedirs(save_dir, exist_ok=True)

            for bal_name, bal_class in BALANCING_METHODS.items():
                counter += 1
                tag = f'{model_name} + {bal_name} ({ds_name})'
                print(f"\n[{counter}/{total}] {tag}")

                # --- Balancing ---
                sampler = bal_class(random_state=SEED)
                X_bal, y_bal = sampler.fit_resample(X_train, y_train_oh)

                history = None
                y_pred_prob = None

                # --- Train ---
                if model_type == 'mlp_baseline':
                    model = create_mlp_baseline(X_bal.shape[1])
                    history = model.fit(X_bal, y_bal, epochs=50, batch_size=32,
                                        validation_split=0.15, callbacks=get_callbacks(), verbose=0)
                    y_pred_prob = model.predict(X_test)
                    y_pred = np.argmax(y_pred_prob, axis=1)
                    del model; gc.collect(); K.clear_session()

                elif model_type == 'mlp_advance':
                    model = create_mlp_advance(X_bal.shape[1])
                    history = model.fit(X_bal, y_bal, epochs=50, batch_size=32,
                                        validation_split=0.15, callbacks=get_callbacks(), verbose=0)
                    y_pred_prob = model.predict(X_test)
                    y_pred = np.argmax(y_pred_prob, axis=1)
                    del model; gc.collect(); K.clear_session()

                elif model_type == 'mlp_tuner':
                    builder = make_tuner_builder(X_bal.shape[1])
                    tuner = kt.RandomSearch(
                        builder, objective='val_accuracy', max_trials=5,
                        executions_per_trial=1,
                        directory=f'tuner_viz_{bal_name}_{ds_name}',
                        project_name='viz', overwrite=True
                    )
                    tuner.search(X_bal, y_bal, epochs=50, batch_size=32,
                                 validation_split=0.15, callbacks=get_callbacks(), verbose=0)
                    best_model = tuner.get_best_models(1)[0]
                    y_pred_prob = best_model.predict(X_test)
                    y_pred = np.argmax(y_pred_prob, axis=1)
                    del best_model, tuner; gc.collect(); K.clear_session()

                else:  # sklearn
                    y_bal_1d = np.argmax(y_bal, axis=1)
                    if model_type == 'naive_bayes':
                        mdl = create_naive_bayes()
                    elif model_type == 'svm':
                        mdl = create_svm(random_state=SEED)
                    elif model_type == 'random_forest':
                        mdl = create_random_forest(random_state=SEED)
                    elif model_type == 'logistic_regression':
                        mdl = create_logistic_regression(random_state=SEED)
                    mdl.fit(X_bal, y_bal_1d)
                    y_pred = mdl.predict(X_test)
                    y_pred_prob = mdl.predict_proba(X_test)

                # --- Panggil fungsi evaluation.py ---
                prefix = f'{save_dir}/{bal_name}'

                print_classification_report(y_true, y_pred, title=tag)

                plot_confusion_matrix(y_true, y_pred, tag,
                                      save_path=f'{prefix}_cm.png')

                if y_pred_prob is not None:
                    plot_roc_curve(y_true, y_pred_prob, tag,
                                   save_path=f'{prefix}_roc.png')

                if history is not None:
                    plot_learning_curve(history, tag,
                                        save_path=f'{prefix}_learning.png')

                print(f"  Acc={accuracy_score(y_true, y_pred):.4f} | F1={f1_score(y_true, y_pred, average='weighted'):.4f}")

    print("\nSelesai!")
    return


@app.cell
def _():
    return


@app.cell
def _():
    #bagian setelah ini tidak perlu di running (old code)
    return


@app.cell
def _(df_fadly, df_rendika, run_all_experiments):
    # Test 1 run
    all_results = run_all_experiments(df_rendika, df_fadly, n_runs=1,seeds=[42], kolom_y_a='Label', kolom_y_b='label')
    return (all_results,)


@app.cell
def _(df_fadly, df_rendika, run_all_experiments):
    # all eksperimen 
    all_results_all = run_all_experiments(df_rendika, df_fadly,kolom_y_a='Label')
    return


@app.cell
def _():
    from src.evaluation import print_all_metrics_table, find_best_model

    return find_best_model, print_all_metrics_table


@app.cell
def _(all_results, find_best_model, print_all_metrics_table):
    #Cetak tabelsemua metrik (accuracy, precision, recall, f1)
    #per dataset + tabel selisih rendika vs fadly
    print_all_metrics_table(all_results)

    #cari model terbaik per dataset
    best = find_best_model(all_results, metric='accuracy')
    best_f1 = find_best_model(all_results, metric='f1')
    return


@app.cell
def _():
    import json
    import numpy as np

    return json, np


@app.cell
def _(json, np):
    def convert_results(obj):
        if isinstance(obj, dict):
            return {k: convert_results(v) for k,v in obj.items()}
        elif isinstance(obj, list):
            return [convert_results(i) for i in obj]
        elif isinstance(obj,(np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return obj


    with open('results/all_results.json)','w') as f:
        json.dump(convert_results(all_results), f,indent=2)

    #untuk load json
    with open('results/all_results.json', 'r') as f:
        all_results =json.load(f)
    return (all_results,)


@app.cell
def _():
    # EValuasi

    import os, gc
    import numpy as np
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, f1_score
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras import backend as K
    import keras_tuner as kt

    from imblearn.over_sampling import RandomOverSampler, SMOTE
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.combine import SMOTEENN,SMOTETomek

    from src.models import (
        create_mlp_baseline, create_mlp_advance, make_tuner_builder,create_naive_bayes, create_logistic_regression, create_random_forest, create_svm, get_callbacks, MODEL_CONFIGS
    )

    from src.balancing import BALANCING_METHODS
    from src.evaluation import (
        plot_confusion_matrix,
        plot_roc_curve,
        plot_learning_curve,
        print_all_metrics_table,
        print_classification_report,
        find_best_model
    )

    SEED=42
    return (
        BALANCING_METHODS,
        K,
        MODEL_CONFIGS,
        SEED,
        TfidfVectorizer,
        accuracy_score,
        create_logistic_regression,
        create_mlp_advance,
        create_mlp_baseline,
        create_naive_bayes,
        create_random_forest,
        create_svm,
        f1_score,
        find_best_model,
        gc,
        get_callbacks,
        kt,
        make_tuner_builder,
        np,
        os,
        pd,
        plot_confusion_matrix,
        plot_learning_curve,
        plot_roc_curve,
        print_all_metrics_table,
        print_classification_report,
        to_categorical,
        train_test_split,
    )


@app.cell
def _(
    SEED,
    TfidfVectorizer,
    df_fadly,
    df_rendika,
    to_categorical,
    train_test_split,
):
    def prepare(df, kx, ky):
        X_tr, X_te, y_tr, y_te = train_test_split(df[kx], df[ky], test_size=0.2, random_state=SEED)

        tfidf = TfidfVectorizer(max_features=5000, min_df=2, max_df=0.95, ngram_range=(1,2))

        return tfidf.fit_transform(X_tr), tfidf.transform(X_te), to_categorical(y_tr,3), to_categorical(y_te,3)

    data_prepared = {
        'Dataset_A': prepare(df_rendika, 'clean_text', 'Label'),
        'Dataset_B': prepare(df_fadly, 'clean_text', 'label'),
    }
    return (data_prepared,)


@app.cell
def _(
    BALANCING_METHODS,
    K,
    MODEL_CONFIGS,
    SEED,
    accuracy_score,
    create_logistic_regression,
    create_mlp_advance,
    create_mlp_baseline,
    create_naive_bayes,
    create_random_forest,
    create_svm,
    data_prepared,
    f1_score,
    gc,
    get_callbacks,
    kt,
    make_tuner_builder,
    np,
    os,
    plot_confusion_matrix,
    plot_learning_curve,
    plot_roc_curve,
    print_classification_report,
):
    DATASET_FOLDERS = {'Dataset_A': 'dataset_rendika', 'Dataset_B': 'dataset_fadly'}
    total = len(data_prepared) * len(BALANCING_METHODS) * len(MODEL_CONFIGS)
    counter = 0

    for ds_name, (X_train, X_test, y_train_oh, y_test_oh) in data_prepared.items():
        y_true = np.argmax(y_test_oh, axis=1)

        for model_name, model_type in MODEL_CONFIGS.items():
            save_dir = f'results/{DATASET_FOLDERS[ds_name]}/{model_name.replace(" ","_")}'
            os.makedirs(save_dir, exist_ok=True)

            for bal_name, bal_class in BALANCING_METHODS.items():
                counter += 1
                tag = f'{model_name} + {bal_name} ({ds_name})'
                print(f"\n[{counter}/{total}] {tag}")

                # --- Balancing ---
                sampler = bal_class(random_state=SEED)
                X_bal, y_bal = sampler.fit_resample(X_train, y_train_oh)

                history = None
                y_pred_prob = None

                # --- Train ---
                if model_type == 'mlp_baseline':
                    model = create_mlp_baseline(X_bal.shape[1])
                    history = model.fit(X_bal, y_bal, epochs=50, batch_size=32,
                                        validation_split=0.15, callbacks=get_callbacks(), verbose=0)
                    y_pred_prob = model.predict(X_test)
                    y_pred = np.argmax(y_pred_prob, axis=1)
                    del model; gc.collect(); K.clear_session()

                elif model_type == 'mlp_advance':
                    model = create_mlp_advance(X_bal.shape[1])
                    history = model.fit(X_bal, y_bal, epochs=50, batch_size=32,
                                        validation_split=0.15, callbacks=get_callbacks(), verbose=0)
                    y_pred_prob = model.predict(X_test)
                    y_pred = np.argmax(y_pred_prob, axis=1)
                    del model; gc.collect(); K.clear_session()

                elif model_type == 'mlp_tuner':
                    builder = make_tuner_builder(X_bal.shape[1])
                    tuner = kt.RandomSearch(
                        builder, objective='val_accuracy', max_trials=5,
                        executions_per_trial=1,
                        directory=f'tuner_viz_{bal_name}_{ds_name}',
                        project_name='viz', overwrite=True
                    )
                    tuner.search(X_bal, y_bal, epochs=50, batch_size=32,
                                 validation_split=0.15, callbacks=get_callbacks(), verbose=0)
                    best_model = tuner.get_best_models(1)[0]
                    y_pred_prob = best_model.predict(X_test)
                    y_pred = np.argmax(y_pred_prob, axis=1)
                    del best_model, tuner; gc.collect(); K.clear_session()

                else:  # sklearn
                    y_bal_1d = np.argmax(y_bal, axis=1)
                    if model_type == 'naive_bayes':
                        mdl = create_naive_bayes()
                    elif model_type == 'svm':
                        mdl = create_svm(random_state=SEED)
                    elif model_type == 'random_forest':
                        mdl = create_random_forest(random_state=SEED)
                    elif model_type == 'logistic_regression':
                        mdl = create_logistic_regression(random_state=SEED)
                    mdl.fit(X_bal, y_bal_1d)
                    y_pred = mdl.predict(X_test)
                    y_pred_prob = mdl.predict_proba(X_test)

                # --- Panggil fungsi evaluation.py ---
                prefix = f'{save_dir}/{bal_name}'

                print_classification_report(y_true, y_pred, title=tag)

                plot_confusion_matrix(y_true, y_pred, tag,
                                      save_path=f'{prefix}_cm.png')

                if y_pred_prob is not None:
                    plot_roc_curve(y_true, y_pred_prob, tag,
                                   save_path=f'{prefix}_roc.png')

                if history is not None:
                    plot_learning_curve(history, tag,
                                        save_path=f'{prefix}_learning.png')

                print(f"  Acc={accuracy_score(y_true, y_pred):.4f} | F1={f1_score(y_true, y_pred, average='weighted'):.4f}")

    print("\nSelesai!")
    return


if __name__ == "__main__":
    app.run()

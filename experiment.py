import marimo

__generated_with = "0.22.0"
app = marimo.App(width="medium", auto_download=["ipynb"])


@app.cell
def _():
    import pandas as pd
    from src.balancing import run_all_experiments

    return pd, run_all_experiments


@app.cell
def _(pd):
    df_rendika = pd.read_csv('data/processed/preprocessed_dataset_rendika.csv')
    df_fadly = pd.read_csv('data/processed/preprocessed_dataset_fadly.csv')
    return df_fadly, df_rendika


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
    return


if __name__ == "__main__":
    app.run()

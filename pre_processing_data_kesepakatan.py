import marimo

__generated_with = "0.22.0"
app = marimo.App(width="medium", auto_download=["ipynb"])


@app.cell
def _():
    import pandas as pd
    from src.labeling import load_kesepakatan_data, show_label_distribution
    from src.preprocessing import full_preprocessing

    return full_preprocessing, load_kesepakatan_data, show_label_distribution


@app.cell
def _(full_preprocessing, load_kesepakatan_data, show_label_distribution):
    df_raw = load_kesepakatan_data('data/raw/data_kesepakatan.csv')
    show_label_distribution(df_raw, 'Dataset Kesepakatan (Sebelum preprocessing)')

    df = full_preprocessing(df_raw, kolom='full_text', use_stemming=True)
    df.to_csv('data/processed/preprocessed_dataset_kesepakatan.csv', index=False)
    show_label_distribution(df, 'Dataset Kesepakatan (Setelah Preprocessing)')
    return


if __name__ == "__main__":
    app.run()

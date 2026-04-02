import marimo

__generated_with = "0.22.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import pandas as pd
    import numpy as np

    return (pd,)


@app.cell
def _(pd):
    df = pd.read_csv('data/raw/rendika_data.csv')
    df = df.drop(columns='Unnamed: 2')
    df
    return (df,)


@app.cell
def _(df):
    #buang text kosong/Nan
    _df = df[df['clean_text'].notna()]
    _df = _df[_df['clean_text'].str.strip() != '']
    _df = _df.reset_index(drop=True)


    print(f'Dataset A -- total: {len(_df)}')
    print(_df['Label'].value_counts().sort_index())

    _df
    #convert_to_csv
    _df.to_csv('rendika_data_proses1.csv', index=False)
    return


@app.cell
def _(pd):
    #EDA sebelum pre processing
    #--- Basic Info ---
    df2 = pd.read_csv('data/proses_1/rendika_data_proses1.csv')
    print(df2.shape)
    print(df2.info())
    print(f"Missing values:\n{df2.isnull().sum()}")
    print(f"Duplikasi: {df2.duplicated(subset='clean_text').sum()}")
    return (df2,)


@app.cell
def _():
    import matplotlib.pyplot as plt
    from wordcloud import WordCloud
    from collections import Counter
    import re

    return WordCloud, plt


@app.cell
def _(df2, plt):
    df3=df2
    df3['char_len'] = df3['clean_text'].str.len()
    df3['word_count'] = df3['clean_text'].str.split().str.len()

    fig,axes = plt.subplots(1,2, figsize=(12,4))
    df3['char_len'].hist(bins=50, ax=axes[0])
    axes[0].set_title('Distribusi Panjang Karakter')
    df3['word_count'].hist(bins=50, ax=axes[1])
    axes[1].set_title('Distribusi Jumlah Kata')
    plt.tight_layout()
    plt.show()
    return (df3,)


@app.cell
def _(df3):
    df4=df3
    print(df4[['char_len','word_count']].describe())
    return (df4,)


@app.cell
def _(WordCloud, df4, plt):
    all_text = ' '.join(df4['clean_text'])
    wc = WordCloud(width=800, height=400, background_color='white').generate(all_text)
    plt.figure(figsize=(10,5))
    plt.imshow(wc,interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud(RaW)')
    plt.show()
    return


@app.cell
def _():
    from src.preprocessing import full_preprocessing

    return (full_preprocessing,)


@app.cell
def _(df4, full_preprocessing):
    #preprocessing
    df_pre = full_preprocessing(df4, 'clean_text', use_stemming=True)
    df_pre.to_csv('preprocessed_dataset_rendika.csv',index=False)
    return (df_pre,)


@app.cell
def _(pd):
    df_pre = pd.read_csv('data/processed/preprocessed_dataset_rendika.csv')
    return (df_pre,)


@app.cell
def _(df_pre):
    df_pre
    return


@app.cell
def _(df_pre):
    #hitung data duplikat
    duplikat_count = df_pre['clean_text'].duplicated().sum()
    print(f"Jumlah data duplikat: {duplikat_count}")
    return


@app.cell
def _(df_pre):
    # Menghitung data null
    null_count = df_pre.isnull().sum()
    print(f"Jumlah data null: {null_count}")
    return


@app.cell
def _(df_pre):
    #hapus duplikat
    df_pre2 = df_pre.drop_duplicates(subset=["clean_text"])
    df_pre2
    return (df_pre2,)


@app.cell
def _(df_pre2):
    df_pre2.to_csv('preprocessed_dataset_rendika.csv', index=False)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #FEATURE EXTRACTION
    """)
    return


@app.cell
def _():
    from src.balancing import prepare_features

    return (prepare_features,)


@app.cell
def _(df_pre2, prepare_features):
    X, y, y_one_hot, tfidf = prepare_features(df_pre2, "Dataset Rendika",kolom_y='Label')
    return


if __name__ == "__main__":
    app.run()

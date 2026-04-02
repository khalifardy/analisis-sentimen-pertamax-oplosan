import marimo

__generated_with = "0.22.0"
app = marimo.App()


@app.cell
def _():
    import pandas as pd
    from sklearn.metrics import cohen_kappa_score

    return cohen_kappa_score, pd


@app.cell
def _(pd):
    df_rendika = pd.read_csv('data/processed/preprocessed_dataset_rendika.csv')
    df_faldy = pd.read_csv('data/processed/preprocessed_dataset_fadly.csv')
    return df_faldy, df_rendika


@app.cell
def _(df_rendika):
    df_rendika
    return


@app.cell
def _(df_faldy):
    df_faldy
    return


@app.cell
def _(df_faldy, df_rendika, pd):
    # merge berdasarkan clean_text untuk hitung agreement

    df_agreement = pd.merge(
        df_rendika[['clean_text', 'Label']].rename(columns={'Label':'label_rendika'}),
        df_faldy[['clean_text', 'label']].rename(columns={'label':'label_rendika'}),
        on='clean_text', how='inner'
    )

    df_agreement.to_csv('data_merge_rendika_fadly.csv',index=False)
    return (df_agreement,)


@app.cell
def _(df_agreement):
    df_agreement
    return


@app.cell
def _(cohen_kappa_score, df_agreement):
    agreement_rate = (df_agreement['label_rendika_x']==df_agreement['label_rendika_y']).mean()

    kappa_score = cohen_kappa_score(df_agreement['label_rendika_x'], df_agreement['label_rendika_y'])

    print(f"Jumlah data yang bisa dibandingkan: {len(df_agreement)}")
    print(f"Agreement Rate: {agreement_rate:.4f} ({agreement_rate*100:2f}%)")
    print(f"Cohen's Kappa: {kappa_score:.4f}")
    return


if __name__ == "__main__":
    app.run()

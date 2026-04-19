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
    df_rendika = pd.read_csv('data/raw/rendika_data_merge.csv')
    df_faldy = pd.read_csv('data/raw/fadly_data_merge.csv')
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
        df_rendika[['full_text', 'label']].rename(columns={'label':'label_rendika'}),
        df_faldy[['full_text', 'label']].rename(columns={'label':'label_fadly'}),
        on='full_text', how='inner'
    )

    df_agreement.insert(0, 'id', range(1,len(df_agreement)+1))
    return (df_agreement,)


@app.cell
def _(df_agreement):
    df_agreement
    return


@app.cell(hide_code=True)
def _():
    def _():
        import marimo as mo
        return


    _()
    return


@app.cell(hide_code=True)
def _(df_agreement, mo):
    _df = mo.sql(
        f"""
        SELECT *
        FROM df_agreement
        WHERE label_rendika != label_fadly
        """
    )
    return


@app.cell
def _(df_agreement):
    df_agreement.to_csv('data/raw/perbandinga.csv', index=False)
    return


@app.cell
def _(cohen_kappa_score, df_agreement):
    agreement_rate = (df_agreement['label_rendika']==df_agreement['label_fadly']).mean()

    kappa_score = cohen_kappa_score(df_agreement['label_rendika'], df_agreement['label_fadly'])

    print(f"Jumlah data yang bisa dibandingkan: {len(df_agreement)}")
    print(f"Agreement Rate: {agreement_rate:.4f} ({agreement_rate*100:2f}%)")
    print(f"Cohen's Kappa: {kappa_score:.4f}")
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()

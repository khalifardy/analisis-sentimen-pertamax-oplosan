import marimo

__generated_with = "0.22.0"
app = marimo.App()


@app.cell
def _():
    import pandas as pd

    return (pd,)


@app.cell
def _(pd):
    df_rendika = pd.read_csv('data/raw/rendika_data.csv')
    df_faldy = pd.read_csv('data/raw/fadly_data.csv')
    df_merge = pd.read_csv('data/raw/mergeddataset.csv')
    return df_faldy, df_merge, df_rendika


@app.cell
def _(df_rendika):
    df_rendika
    return


@app.cell
def _(df_faldy):
    df_faldy
    return


@app.cell
def _(df_merge):
    df_merge
    return


@app.cell
def _(df_merge, df_rendika, pd):
    perbandingan = pd.concat([df_merge,df_rendika], axis=1)
    perbandingan[['full_text','clean_text']]
    return


@app.cell
def _(df_merge):
    df_merge2 = df_merge[df_merge['full_text']!='@moneyfestinglux topiknya klasemen liga korupsi.']

    df_merge2
    return (df_merge2,)


@app.cell
def _(df_merge2, df_rendika, pd):
    df_merge_rendika = pd.concat([df_merge2,df_rendika[['Label']]], axis=1)
    df_merge_rendika
    return (df_merge_rendika,)


@app.cell
def _(df_merge_rendika):
    df_merge_rendika.to_csv('data/raw/rendika_data_merge.csv', index=False)
    return


@app.cell
def _(df_faldy, df_merge2, pd):
    df_merge_fadly = pd.concat([df_merge2,df_faldy[['label']]], axis=1)
    df_merge_fadly
    return (df_merge_fadly,)


@app.cell
def _(df_merge_fadly):
    df_merge_fadly.to_csv('data/raw/fadly_data_merge.csv', index=False)
    return


if __name__ == "__main__":
    app.run()

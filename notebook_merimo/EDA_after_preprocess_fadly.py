import marimo

__generated_with = "0.22.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    return pd, plt


@app.cell
def _(pd):
    df = pd.read_csv('data/processed/preprocessed_dataset_fadly.csv')
    return (df,)


@app.cell
def _(df):
    #distribusi label
    print('\n Distribusi label')
    print(df['label'].value_counts())
    print(f'\n Presentase')
    print(df['label'].value_counts(normalize=True))
    return


@app.cell
def _(df):
    df
    return


@app.cell
def _(df):
    df2=df
    df2['label'] = df['label'].map(
        {0:'negatif',1:'netral',2:'positif'}
    )
    return (df2,)


@app.cell
def _(df2, plt):
    #fig,axes = plt.subplots(1,2,figsize=(14,5))

    #Pie Charts distribusi label
    colors = {
        'positif': '#2ecc71',
        'negatif': '#e74c3c',
        'netral': '#3498db'
    }

    label_counts = df2['label'].value_counts()
    plt.pie(label_counts.values, labels=label_counts.index, autopct='%1.1f%%',
                colors=[colors[label] for label in label_counts.index])
    plt.title('Distrbusi Sentiment')

    plt.tight_layout()
    plt.savefig('distribusi_label.png', dpi=300)
    plt.show()
    return


@app.cell
def _(df2):
    #panjang text per sentimen
    df_only_text_clean = df2
    df_only_text_clean['text_length'] = df_only_text_clean['clean_text'].apply(len)
    df_only_text_clean.groupby('label')['text_length'].mean()
    return (df_only_text_clean,)


@app.cell
def _():
    from wordcloud import WordCloud

    return (WordCloud,)


@app.cell
def _(WordCloud, df_only_text_clean, plt):
    text = ' '.join(df_only_text_clean['clean_text'].astype(str))

    #buat wordcloud

    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
    ).generate(text)

    plt.figure(figsize=(10,5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()
    return


@app.cell
def _():
    from matplotlib import colormaps

    return


@app.cell
def _(WordCloud, df_only_text_clean, plt):
    #sentimen negatif (label = 0)
    text_negatif = ' '.join(df_only_text_clean[df_only_text_clean['label']=='negatif']['clean_text'].astype(str))
    wc_negatif = WordCloud(
        width=800,
        height=400,
        background_color='white',
        colormap = 'Reds'
    ).generate(text_negatif)

    text_netral = ' '.join(df_only_text_clean[df_only_text_clean['label']=='netral']['clean_text'].astype(str))
    wc_netral = WordCloud(
        width=800,
        height=400,
        background_color='white',
        colormap = 'Blues'
    ).generate(text_netral)

    text_positif = ' '.join(df_only_text_clean[df_only_text_clean['label']=='positif']['clean_text'].astype(str))
    wc_positif = WordCloud(
        width=800,
        height=400,
        background_color='white',
        colormap = 'Greens'
    ).generate(text_positif)


    fig, axes = plt.subplots(1,3, figsize=(18,5))

    axes[0].imshow(wc_negatif, interpolation='bilinear')
    axes[0].set_title('Negatif')
    axes[0].axis('off')

    axes[1].imshow(wc_netral, interpolation='bilinear')
    axes[1].set_title('Netral')
    axes[1].axis('off')

    axes[2].imshow(wc_positif, interpolation='bilinear')
    axes[2].set_title('Positif')
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()
    return


if __name__ == "__main__":
    app.run()

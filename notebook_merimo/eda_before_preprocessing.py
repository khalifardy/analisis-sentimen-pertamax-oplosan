import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell
def _():
    return


@app.cell
def _():
    import pandas as pd
    import matplotlib.pyplot as plt
    from wordcloud import WordCloud
    from collections import Counter
    import re

    return WordCloud, pd, plt, re


@app.cell
def _(pd):
    df = pd.read_csv('data/raw/mergeddataset.csv')
    df
    return (df,)


@app.cell
def _(df):
    #--- Basic Info ---
    print(df.shape)
    print(df.info())
    print(f"Missing values:\n{df.isnull().sum()}")
    print(f"Duplikasi: {df.duplicated(subset='full_text').sum()}")
    return


@app.cell
def _(df, plt):
    #--- statistik teks ---
    df2 = df
    df2['char_len'] = df2['full_text'].str.len()
    df2['word_count'] = df2['full_text'].str.split().str.len()

    fig,axes = plt.subplots(1,2, figsize=(12,4))
    df2['char_len'].hist(bins=50, ax=axes[0])
    axes[0].set_title('Distribusi Panjang Karakter')
    df2['word_count'].hist(bins=50, ax=axes[1])
    axes[1].set_title('Distribusi Jumlah Kata')
    plt.tight_layout()
    plt.show()
    return (df2,)


@app.cell
def _(df2):
    print(df2[['char_len', 'word_count']].describe())
    return


@app.cell
def _(WordCloud, df2, plt):
    all_text = ' '.join(df2['full_text'].dropna())
    wc = WordCloud(width=800, height=400, background_color='white').generate(all_text)
    plt.figure(figsize=(10,5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud(Raw)')
    plt.show()
    return


@app.cell
def _(df2, plt):
    #----Top Hashtags----
    hashtags = df2['full_text'].str.findall(r'#\w+').explode().dropna()
    hashtags.value_counts().head(15).plot(kind='barh', title='Top 15 Hashtags')
    plt.show()
    return


@app.cell
def _(df2, plt):
    #--- Top Mentions ---
    mentions = df2['full_text'].str.findall(r'@\w+').explode().dropna()
    mentions.value_counts().head(15).plot(kind='barh', title='Top 15 Mentions')
    plt.show()
    return


@app.cell
def _(df2, re):
    #--- Proporsi Noise ----
    df3=df2
    df3['has_url'] = df3['full_text'].str.contains(r'http\S+',na=False)
    df3['has_emoji'] = df3['full_text'].str.contains(r'[\U00010000-\U0010ffff]', na=False, flags=re.UNICODE)
    df3['is_retweet'] = df3['full_text'].str.startswith('RT @', na=False)

    print(f"Mengandung URL: {df3['has_url'].mean():.1%}")
    print(f"Mengandung Emoji: {df3['has_emoji'].mean():.1%}")
    print(f"Merupakan Retweet: {df3['is_retweet'].mean():.1%}")
    return (df3,)


@app.cell
def _(df3):
    #----- Tweet terlalu pendek (mungkin tidak informatif) ---

    short_tweets = df3[df3['word_count'] <= 3]
    print(f"Tweet ≤ 3 kata: {len(short_tweets)} ({len(short_tweets)/len(df3):.1%})")
    print(short_tweets['full_text'].head(10))
    return


if __name__ == "__main__":
    app.run()

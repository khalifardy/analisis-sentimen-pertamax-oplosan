# Analisis Sentimen Opini Publik terhadap Pertamax Oplosan

**Tugas Akhir** - Program Studi Informatika

| | |
|---|---|
| **Nama** | Rendika Tahir Ahmad |
| **NIM** | 1304211041 |
| **Topik** | Analisis Sentimen Opini Publik terhadap Pertamax Oplosan Menggunakan Multi-Layer Perceptron dan Machine Learning Tradisional |

## Deskripsi

Penelitian ini menganalisis sentimen opini publik di media sosial Twitter/X terkait isu **Pertamax oplosan** di Indonesia. Data dikumpulkan menggunakan teknik scraping, kemudian dilabeli secara manual oleh **2 annotator berbeda** ke dalam 3 kategori: **Positif**, **Netral**, dan **Negatif**.

Kedua dataset annotator diproses secara terpisah melalui pipeline yang sama, kemudian hasilnya dibandingkan untuk melihat konsistensi dan pengaruh perbedaan labeling terhadap performa model.

## Metodologi

### Pipeline

1. **Data Collection** - Scraping data Twitter menggunakan snscrape
2. **Exploratory Data Analysis** - Statistik teks, word cloud, top hashtags & mentions
3. **Text Preprocessing** - Case folding, cleansing, normalisasi slang, stopword removal, stemming (Sastrawi)
4. **Labeling** - Manual oleh 2 annotator berbeda + Inter-Annotator Agreement (Cohen's Kappa)
5. **Feature Extraction** - TF-IDF Vectorization (max_features=5000, ngram 1-2)
6. **Data Balancing** - 5 metode: ROS, SMOTE, RUS, SMOTEENN, SMOTETomek
7. **Modeling** - 7 model (3 MLP + 4 ML tradisional)
8. **Evaluasi** - Setiap kombinasi dijalankan **3 kali** lalu dirata-ratakan

### Model yang Digunakan

| Kategori | Model |
|----------|-------|
| Deep Learning | MLP Baseline, MLP Advanced (BatchNorm + Dropout), MLP + Keras Tuner |
| ML Tradisional | Naive Bayes, SVM (Linear), Random Forest, Logistic Regression |

### Metrik Evaluasi

- Accuracy, Precision, Recall, F1-Score (weighted)
- Confusion Matrix (normal & normalized)
- ROC Curve & AUC
- Learning Curve (untuk model MLP)

## Struktur Proyek

```
analisis-sentimen-pertamax-oplosan/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/                    # Data mentah dari 2 annotator
│   │   ├── data_annotator1.csv
│   │   └── data_annotator2_fadly.csv
│   └── processed/              # Data hasil preprocessing
├── notebooks/
│   └── analisis_sentimen.ipynb # Notebook utama
├── src/
│   ├── __init__.py
│   ├── preprocessing.py        # Fungsi text preprocessing
│   ├── labeling.py             # Load data, cleaning label, agreement
│   ├── balancing.py            # Metode balancing + multi-run
│   ├── models.py               # Definisi model MLP & sklearn
│   └── evaluation.py           # Evaluasi, visualisasi, tabel
├── results/
│   ├── dataset_a/              # Hasil evaluasi dataset annotator 1
│   └── dataset_b/              # Hasil evaluasi dataset annotator 2
└── docs/
    └── revisi_pembimbing.md    # Catatan revisi dari pembimbing
```

## Instalasi

```bash
# Clone repository
git clone https://github.com/USERNAME/analisis-sentimen-pertamax-oplosan.git
cd analisis-sentimen-pertamax-oplosan

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt')"
```

## Cara Menjalankan

### Opsi 1: Notebook (Rekomendasi)
```bash
jupyter notebook notebooks/analisis_sentimen.ipynb
```

### Opsi 2: Script Python
```python
from src.labeling import load_and_clean_data
from src.preprocessing import full_preprocessing
from src.balancing import run_all_experiments
from src.evaluation import print_comparison_table

# Load data
df_a, df_b = load_and_clean_data()

# Preprocessing (jika data belum di-preprocess)
# df_a = full_preprocessing(df_a, 'clean_text')
# df_b = full_preprocessing(df_b, 'clean_text')

# Jalankan semua eksperimen
results = run_all_experiments(df_a, df_b)

# Tampilkan perbandingan
print_comparison_table(results)
```

## Hasil

> Bagian ini akan diisi setelah semua eksperimen selesai dijalankan.

## Teknologi

- Python 3.10+
- TensorFlow / Keras
- scikit-learn
- imbalanced-learn
- Sastrawi (stemmer Bahasa Indonesia)
- NLTK

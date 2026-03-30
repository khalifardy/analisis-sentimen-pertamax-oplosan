"""
Text Preprocessing Module
=========================
Fungsi-fungsi untuk preprocessing teks bahasa Indonesia:
- Case folding
- Cleansing (hapus mention, hashtag, URL, RT, angka, emoji, tanda baca)
- Normalisasi slang
- Stopword removal (custom + Sastrawi)
- Stemming (Sastrawi)
"""

import re
import string
import nltk
from nltk.tokenize import word_tokenize
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Download NLTK data jika belum ada
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)


# ============= CUSTOM STOPWORDS INDONESIA =============
CUSTOM_STOPWORDS = {
    'yg', 'yang', 'dan', 'di', 'ini', 'itu', 'dengan', 'untuk', 'pada', 'ke',
    'dari', 'adalah', 'akan', 'juga', 'sudah', 'bisa', 'ada', 'tidak', 'saya',
    'kamu', 'mereka', 'kita', 'apa', 'atau', 'jika', 'bila', 'saat', 'ketika',
    'lebih', 'sangat', 'hanya', 'banyak', 'seperti', 'karena', 'oleh', 'dalam',
    'nya', 'lah', 'kan', 'dong', 'deh', 'sih', 'nih', 'tuh', 'kok', 'ya',
    'oh', 'ah', 'eh', 'aja', 'sama', 'jadi', 'udah', 'gak', 'ga', 'mau',
    'buat', 'tapi', 'kalau', 'kalo', 'masih', 'lagi', 'biar', 'dulu', 'nih',
    'https', 'http', 'www', 'co', 'rt', 'amp',
    't', 's', 'd', 'n', 'x'
}


# ============= KAMUS SLANG INDONESIA =============
SLANG_DICT = {
    'gak': 'tidak', 'ga': 'tidak', 'gk': 'tidak', 'ngga': 'tidak', 'nggak': 'tidak',
    'gue': 'saya', 'gw': 'saya', 'gua': 'saya', 'aku': 'saya', 'ak': 'saya',
    'lu': 'kamu', 'lo': 'kamu', 'elu': 'kamu', 'loe': 'kamu',
    'yg': 'yang', 'dgn': 'dengan', 'utk': 'untuk', 'tdk': 'tidak',
    'bgt': 'banget', 'bngt': 'banget', 'bngtt': 'banget',
    'emg': 'memang', 'emang': 'memang',
    'udh': 'sudah', 'udah': 'sudah', 'sdh': 'sudah',
    'blm': 'belum', 'blum': 'belum',
    'org': 'orang', 'orng': 'orang',
    'krn': 'karena', 'karna': 'karena', 'krna': 'karena',
    'sm': 'sama', 'sma': 'sama',
    'tp': 'tapi', 'tpi': 'tapi',
    'jg': 'juga', 'jga': 'juga',
    'bkn': 'bukan', 'bukn': 'bukan',
    'klo': 'kalau', 'kalo': 'kalau', 'klw': 'kalau',
    'aja': 'saja', 'aj': 'saja',
    'dpt': 'dapat', 'bs': 'bisa', 'bsa': 'bisa',
    'gmn': 'bagaimana', 'gimana': 'bagaimana',
    'knp': 'kenapa', 'knpa': 'kenapa',
    'skrg': 'sekarang', 'skrang': 'sekarang',
    'hrs': 'harus', 'hrus': 'harus',
    'trs': 'terus', 'trus': 'terus',
    'btw': 'ngomong-ngomong',
    'gpp': 'tidak apa-apa', 'gapapa': 'tidak apa-apa',
    'makasih': 'terima kasih', 'makasi': 'terima kasih', 'thx': 'terima kasih',
    'pls': 'tolong', 'plz': 'tolong', 'plis': 'tolong',
    'banget': 'sangat', 'bener': 'benar', 'bnr': 'benar',
    'cmn': 'cuman', 'cuma': 'hanya', 'cuman': 'hanya',
    'dah': 'sudah', 'deh': 'deh', 'dong': 'dong', 'donk': 'dong',
    'sih': 'sih', 'nih': 'ini', 'tuh': 'itu',
    'wkwk': '', 'wkwkwk': '', 'haha': '', 'hahaha': '', 'hihi': '',
    'anjir': '', 'anjay': '', 'anjg': '', 'bgst': '', 'bngst': '',
}


# ============= CLASS SASTRAWI =============
class SastrawiPreprocessor:
    """Wrapper untuk Sastrawi stopword remover dan stemmer."""

    def __init__(self):
        self.factory_stopword = StopWordRemoverFactory()
        self.stopword_remover = self.factory_stopword.create_stop_word_remover()
        self.factory_stemmer = StemmerFactory()
        self.stemmer = self.factory_stemmer.create_stemmer()

    def remove_stopwords(self, text):
        return self.stopword_remover.remove(text)

    def stemming(self, text):
        return self.stemmer.stem(text)


# ============= FUNGSI PREPROCESSING =============

def case_folding(text):
    """Mengubah semua huruf menjadi lowercase."""
    return text.lower()


def cleansing(text):
    """Membersihkan karakter yang tidak diperlukan."""
    # Hapus URL (termasuk t.co)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r't\.co/\S+', '', text)
    # Hapus mention
    text = re.sub(r'@\w+', '', text)
    # Hapus hashtag
    text = re.sub(r'#\w+', '', text)
    # Hapus RT
    text = re.sub(r'\bRT\b', '', text, flags=re.IGNORECASE)
    # Hapus angka
    text = re.sub(r'\d+', '', text)
    # Hapus emoji dan karakter unicode
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    # Hapus tanda baca
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Hapus whitespace berlebih
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def remove_custom_stopwords(text, stopwords_set=None):
    """Hapus custom stopwords dari teks."""
    if stopwords_set is None:
        stopwords_set = CUSTOM_STOPWORDS
    words = text.split()
    filtered = [word for word in words if word.lower() not in stopwords_set and len(word) > 2]
    return ' '.join(filtered)


def normalize_slang(text, slang_dict=None):
    """Normalisasi kata-kata slang/informal."""
    if slang_dict is None:
        slang_dict = SLANG_DICT
    words = text.split()
    normalized = [slang_dict.get(word, word) for word in words]
    return ' '.join(normalized)


def tokenize(text):
    """Tokenisasi teks."""
    return word_tokenize(text)


def full_preprocessing(dataframe, kolom, use_stemming=True):
    """
    Pipeline preprocessing lengkap.

    Parameters:
    - dataframe: pandas DataFrame
    - kolom: nama kolom yang berisi teks
    - use_stemming: boolean, apakah menggunakan stemming

    Returns:
    - dataframe dengan kolom 'clean_text' baru
    """
    sastrawi = SastrawiPreprocessor()

    print("....mulai case folding...")
    dataframe['clean_text'] = dataframe[kolom].apply(case_folding)

    print("....mulai cleansing...")
    dataframe['clean_text'] = dataframe['clean_text'].apply(cleansing)

    print("....mulai normalize slang...")
    dataframe['clean_text'] = dataframe['clean_text'].apply(
        lambda x: normalize_slang(x, SLANG_DICT)
    )

    print("....mulai remove stopword...")
    dataframe['clean_text'] = dataframe['clean_text'].apply(
        sastrawi.remove_stopwords
    )

    if use_stemming:
        print("....mulai stemming...")
        dataframe['clean_text'] = dataframe['clean_text'].apply(
            sastrawi.stemming
        )

    # Hapus teks kosong setelah preprocessing
    dataframe = dataframe[dataframe['clean_text'].notna()]
    dataframe = dataframe[dataframe['clean_text'].str.strip() != '']
    dataframe = dataframe.reset_index(drop=True)

    print(f"Preprocessing selesai. Total data: {len(dataframe)}")
    return dataframe

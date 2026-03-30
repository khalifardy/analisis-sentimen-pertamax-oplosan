# Catatan Revisi dari Pembimbing

## Revisi 1: Tabel 4.4.5 Bab 4
**Masalah**: Yang dibandingkan di tabel bukan data training vs data testing.
**Perbaikan**: Yang ditampilkan adalah **persentase kategori positif, negatif, netral** dari masing-masing dataset.

## Revisi 2: Evaluasi 3x Run
**Masalah**: Evaluasi hanya dilakukan 1 kali.
**Perbaikan**: Semua 5 metode balancing dijalankan **3 kali** dengan random seed berbeda (42, 123, 456), lalu hasilnya **dirata-ratakan** untuk setiap metrik (accuracy, precision, recall, f1).

## Revisi 3: Labeling Manual
**Masalah**: Labeling menggunakan IndoBERT (model pre-trained).
**Perbaikan**: Labeling dilakukan **secara manual** oleh **2 annotator berbeda**. Kedua dataset diproses secara terpisah melalui pipeline yang sama, lalu hasilnya dibandingkan. Inter-Annotator Agreement dihitung menggunakan Cohen's Kappa.

import pandas as pd
from src.preprocessing import full_preprocessing

def main():
    # 1. Membaca data mentah milikmu yang ada nama EXPORT-nya
    print("Membaca data mentah (data_kesepakatan_export.csv)...")
    try:
        df_raw = pd.read_csv('data_kesepakatan_export.csv')
    except FileNotFoundError:
        print("\n[ERROR] File 'data_kesepakatan_export.csv' TIDAK DITEMUKAN!")
        print("Pastikan file tersebut berada di satu folder yang sama dengan script ini.")
        return

    # 2. Menjalankan fungsi preprocessing dengan Stemming Sastrawi dari folder src milikmu
    print("\nMenjalankan full_preprocessing dengan Stemming Sastrawi...")
    print("Proses ini memakan waktu beberapa menit karena ukuran dataset yang besar. Mohon tunggu...")
    
    try:
        # Membaca kolom 'clean_text' sesuai dengan struktur file export kamu
        df = full_preprocessing(df_raw, kolom='clean_text', use_stemming=True)
    except Exception as e:
        print(f"\n[ERROR] Terjadi kesalahan saat menjalankan preprocessing: {str(e)}")
        return
    
    # 3. Menyimpan hasil bersih ke destinasi folder processed sesuai targetmu
    output_path = r'C:\Users\Asus\Documents\Kuliah\9\Akhir\Codes\analisis-sentimen-pertamax-oplosan\data\processed\preprocessed_dataset_kesepakatan.csv'
    
    try:
        df.to_csv(output_path, index=False)
        print("\n=======================================================")
        print("SUKSES! File preprocessing berhasil disimpan di:")
        print(f"{output_path}")
        print("=======================================================")
    except Exception as e:
        print(f"\n[ERROR] Gagal menyimpan file ke lokasi tujuan: {str(e)}")

if __name__ == "__main__":
    main()

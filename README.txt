# Aplikasi Klasifikasi Berita

Aplikasi Streamlit untuk klasifikasi berita menggunakan:

- Word2Vec Skip-Gram
- MinMaxScaler
- Gaussian Naive Bayes
- Sastrawi untuk preprocessing Bahasa Indonesia

## Struktur folder

```text
aplikasi_berita/
├── app.py
├── model_skipgram_naive_bayes (1).pkl
└── requirements.txt
```

## Instalasi di VS Code

Buka Terminal VS Code pada folder project, lalu:

```bash
pip install -r requirements.txt
```

Jika menggunakan virtual environment:

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

Lalu:

```bash
pip install -r requirements.txt
```

## Menjalankan aplikasi

```bash
streamlit run app.py
```

Aplikasi akan membuka halaman Streamlit di browser.

## Cara menggunakan

1. Masukkan URL berita.
2. Klik **Klasifikasikan Berita**.
3. Aplikasi mengambil isi berita.
4. Teks diproses dengan preprocessing.
5. Teks diubah menjadi document vector menggunakan Skip-Gram.
6. Vector diproses dengan MinMaxScaler.
7. Gaussian Naive Bayes menentukan kategori.
8. Hasil ditampilkan sebagai `sport` atau `finance` beserta confidence.

import streamlit as st
import pickle
import re
import requests
import numpy as np

from bs4 import BeautifulSoup
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory


# ====================================================
# KONFIGURASI
# ====================================================

st.set_page_config(
    page_title="Klasifikasi Berita",
    page_icon="📰",
    layout="wide"
)


# ====================================================
# LOAD MODEL
# ====================================================

MODEL_PATH = "model_skipgram_naive_bayes (1).pkl"

try:
    with open(MODEL_PATH, "rb") as file:
        saved_model = pickle.load(file)

    skipgram_model = saved_model["skipgram_model"]
    scaler = saved_model["scaler"]
    naive_bayes = saved_model["naive_bayes"]

    vector_size = saved_model["vector_size"]
    window = saved_model["window"]
    epochs = saved_model["epochs"]

except Exception as e:
    st.error(
        f"Model gagal dimuat.\n\n"
        f"Error: {e}"
    )
    st.stop()


# ====================================================
# STEMMER
# ====================================================

factory = StemmerFactory()
stemmer = factory.create_stemmer()


# ====================================================
# STOPWORD
# ====================================================

STOPWORDS = {
    "yang", "dan", "di", "ke", "dari", "untuk", "dengan",
    "ini", "itu", "pada", "adalah", "akan", "dalam",
    "juga", "karena", "oleh", "sebagai", "atau", "tersebut",
    "bahwa", "saat", "lebih", "sudah", "telah", "bagi",
    "agar", "hingga", "sehingga", "setelah", "sebelum",
    "dapat", "bisa", "tidak", "ada", "mereka", "kami",
    "kita", "anda", "ia", "dia", "nya", "para", "sebuah",
    "seorang", "terhadap", "menjadi", "masih", "hanya",
    "sangat", "jika", "ketika", "dengan", "namun", "tetapi",
    "secara", "yakni", "yakni", "bahkan", "salah", "satu"
}


# ====================================================
# PREPROCESSING
# ====================================================

def clean_text(text):
    """
    Membersihkan teks dari URL, HTML, angka,
    tanda baca, dan karakter yang tidak diperlukan.
    """

    text = str(text)

    # lowercase
    text = text.lower()

    # hapus URL
    text = re.sub(
        r"http\S+|www\S+|https\S+",
        " ",
        text
    )

    # hapus HTML
    text = re.sub(
        r"<.*?>",
        " ",
        text
    )

    # hapus angka
    text = re.sub(
        r"\d+",
        " ",
        text
    )

    # hanya huruf
    text = re.sub(
        r"[^a-zA-Z\s]",
        " ",
        text
    )

    # hapus spasi berlebih
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


def tokenize(text):
    """
    Tokenisasi teks.
    """

    return text.split()


def remove_stopwords(tokens):
    """
    Menghapus stopword.
    """

    return [
        token
        for token in tokens
        if token not in STOPWORDS
    ]


def stemming(tokens):
    """
    Stemming menggunakan Sastrawi.
    """

    return [
        stemmer.stem(token)
        for token in tokens
    ]


def preprocess_text(text):
    """
    Pipeline preprocessing:
    cleaning → tokenisasi → stopword removal → stemming
    """

    cleaned = clean_text(text)

    tokens = tokenize(cleaned)

    tokens_without_stopword = remove_stopwords(tokens)

    stemmed_tokens = stemming(tokens_without_stopword)

    return stemmed_tokens


# ====================================================
# KEYWORD SPORT
# ====================================================

SPORT_KEYWORDS = {
    "olahraga",
    "sport",
    "sepak",
    "bola",
    "futsal",
    "basket",
    "voli",
    "badminton",
    "bulu",
    "tangkis",
    "tenis",
    "atlet",
    "atletik",
    "lari",
    "renang",
    "tinju",
    "petinju",
    "gulat",
    "balap",
    "motor",
    "motogp",
    "formula",
    "pemain",
    "pelatih",
    "tim",
    "klub",
    "liga",
    "turnamen",
    "kompetisi",
    "pertandingan",
    "laga",
    "gol",
    "menang",
    "kalah",
    "juara",
    "skor",
    "piala",
    "olimpiade",
    "medali",
    "persib",
    "persija",
    "persebaya",
    "pssi",
    "premier",
    "league",
    "champions",
    "uefa",
    "fifa",
    "nba",
    "nfl",
    "mlb",
    "ufc",
    "formula",
    "moto"
}


# ====================================================
# KEYWORD FINANCE
# ====================================================

FINANCE_KEYWORDS = {
    "keuangan",
    "finance",
    "finansial",
    "ekonomi",
    "ekonom",
    "bisnis",
    "investasi",
    "investor",
    "saham",
    "obligasi",
    "reksadana",
    "bank",
    "perbankan",
    "bursa",
    "pasar",
    "modal",
    "rupiah",
    "dolar",
    "usd",
    "valuta",
    "kurs",
    "inflasi",
    "deflasi",
    "suku",
    "bunga",
    "pajak",
    "pendapatan",
    "laba",
    "rugi",
    "profit",
    "perusahaan",
    "emiten",
    "dividen",
    "aset",
    "utang",
    "kredit",
    "pinjaman",
    "devisa",
    "ekspor",
    "impor",
    "produksi",
    "konsumsi",
    "moneter",
    "fiskal",
    "dagang",
    "perdagangan",
    "startup",
    "ipo",
    "market",
    "stock",
    "trading",
    "fund",
    "cryptocurrency",
    "kripto"
}


# ====================================================
# STRONG KEYWORD SPORT
# ====================================================

STRONG_SPORT_KEYWORDS = {
    "sepak",
    "bola",
    "futsal",
    "basket",
    "voli",
    "badminton",
    "tangkis",
    "tenis",
    "atlet",
    "atletik",
    "petinju",
    "tinju",
    "gulat",
    "motogp",
    "formula",
    "pemain",
    "pelatih",
    "tim",
    "klub",
    "liga",
    "turnamen",
    "pertandingan",
    "laga",
    "gol",
    "piala",
    "olimpiade",
    "medali",
    "pssi",
    "fifa",
    "uefa",
    "nba",
    "nfl",
    "mlb",
    "ufc"
}


# ====================================================
# STRONG KEYWORD FINANCE
# ====================================================

STRONG_FINANCE_KEYWORDS = {
    "saham",
    "obligasi",
    "reksadana",
    "investasi",
    "investor",
    "bursa",
    "bank",
    "perbankan",
    "rupiah",
    "dolar",
    "kurs",
    "inflasi",
    "deflasi",
    "suku",
    "bunga",
    "pajak",
    "laba",
    "rugi",
    "profit",
    "emiten",
    "dividen",
    "aset",
    "utang",
    "kredit",
    "pinjaman",
    "moneter",
    "fiskal",
    "ipo",
    "trading",
    "market",
    "stock",
    "cryptocurrency",
    "kripto"
}


# ====================================================
# URL SCRAPING
# ====================================================

def scrape_url(url):
    """
    Mengambil teks berita dari URL.
    """

    try:

        headers = {
            "User-Agent": (
                "Mozilla/5.0 "
                "(Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 "
                "(KHTML, like Gecko) "
                "Chrome/120.0 Safari/537.36"
            )
        }

        response = requests.get(
            url,
            headers=headers,
            timeout=15
        )

        response.raise_for_status()

        soup = BeautifulSoup(
            response.text,
            "html.parser"
        )

        # hapus elemen yang tidak diperlukan
        for element in soup([
            "script",
            "style",
            "nav",
            "footer",
            "header",
            "aside"
        ]):
            element.decompose()

        paragraphs = soup.find_all("p")

        text = " ".join(
            p.get_text(" ", strip=True)
            for p in paragraphs
        )

        return text.strip()

    except Exception as e:

        st.error(
            f"Gagal mengambil berita dari URL.\n\n"
            f"Error: {e}"
        )

        return ""


# ====================================================
# DOCUMENT VECTOR
# ====================================================

def document_vector(tokens, model):
    """
    Mengubah dokumen menjadi rata-rata vektor Word2Vec.
    """

    vectors = []

    for token in tokens:

        if token in model.wv:

            vectors.append(
                model.wv[token]
            )

    if len(vectors) == 0:

        return np.zeros(
            model.vector_size
        )

    return np.mean(
        vectors,
        axis=0
    )


# ====================================================
# VOCABULARY COVERAGE
# ====================================================

def calculate_vocabulary_coverage(tokens, model):

    if len(tokens) == 0:

        return 0.0

    known_tokens = [
        token
        for token in tokens
        if token in model.wv
    ]

    coverage = (
        len(known_tokens)
        / len(tokens)
    ) * 100

    return coverage


# ====================================================
# DOMAIN MATCH
# ====================================================

def get_domain_matches(tokens):

    token_set = set(tokens)

    sport_matches = (
        token_set.intersection(
            SPORT_KEYWORDS
        )
    )

    finance_matches = (
        token_set.intersection(
            FINANCE_KEYWORDS
        )
    )

    strong_sport_matches = (
        token_set.intersection(
            STRONG_SPORT_KEYWORDS
        )
    )

    strong_finance_matches = (
        token_set.intersection(
            STRONG_FINANCE_KEYWORDS
        )
    )

    return (
        sport_matches,
        finance_matches,
        strong_sport_matches,
        strong_finance_matches
    )


# ====================================================
# DOMAIN DETECTION
# ====================================================

def detect_domain(tokens):

    (
        sport_matches,
        finance_matches,
        strong_sport_matches,
        strong_finance_matches
    ) = get_domain_matches(tokens)

    sport_score = len(
        sport_matches
    )

    finance_score = len(
        finance_matches
    )

    strong_sport_score = len(
        strong_sport_matches
    )

    strong_finance_score = len(
        strong_finance_matches
    )

    # Tidak ada keyword Sport maupun Finance
    if (
        sport_score == 0
        and finance_score == 0
    ):

        return (
            "unknown",
            sport_matches,
            finance_matches,
            strong_sport_matches,
            strong_finance_matches
        )

    # Strong Sport
    if (
        strong_sport_score >= 1
        and strong_sport_score > strong_finance_score
    ):

        return (
            "sport",
            sport_matches,
            finance_matches,
            strong_sport_matches,
            strong_finance_matches
        )

    # Strong Finance
    if (
        strong_finance_score >= 1
        and strong_finance_score > strong_sport_score
    ):

        return (
            "finance",
            sport_matches,
            finance_matches,
            strong_sport_matches,
            strong_finance_matches
        )

    # Sport berdasarkan keyword umum
    if (
        sport_score >= 2
        and sport_score > finance_score
    ):

        return (
            "sport",
            sport_matches,
            finance_matches,
            strong_sport_matches,
            strong_finance_matches
        )

    # Finance berdasarkan keyword umum
    if (
        finance_score >= 2
        and finance_score > sport_score
    ):

        return (
            "finance",
            sport_matches,
            finance_matches,
            strong_sport_matches,
            strong_finance_matches
        )

    # Jika ambigu
    return (
        "unknown",
        sport_matches,
        finance_matches,
        strong_sport_matches,
        strong_finance_matches
    )


# ====================================================
# JUDUL APLIKASI
# ====================================================

st.title(
    "📰 Klasifikasi Berita"
)

st.write(
    "Klasifikasi berita menggunakan "
    "**Word2Vec Skip-Gram + Gaussian Naive Bayes**."
)

st.write(
    "Kategori yang tersedia: **Sport** dan **Finance**."
)


# ====================================================
# INFORMASI MODEL
# ====================================================

with st.expander(
    "ℹ️ Informasi Model"
):

    st.write(
        f"**Word2Vec:** Skip-Gram"
    )

    st.write(
        f"**Vector Size:** {vector_size}"
    )

    st.write(
        f"**Window:** {window}"
    )

    st.write(
        f"**Epochs:** {epochs}"
    )

    st.write(
        "**Classifier:** Gaussian Naive Bayes"
    )


# ====================================================
# INPUT
# ====================================================

st.subheader(
    "🔎 Masukkan Berita"
)

input_type = st.radio(
    "Pilih sumber berita:",
    [
        "URL",
        "Teks Berita"
    ],
    horizontal=True
)


url = ""
text_input = ""


if input_type == "URL":

    url = st.text_input(
        "Masukkan URL berita:",
        placeholder="https://contoh.com/berita"
    )

else:

    text_input = st.text_area(
        "Masukkan teks berita:",
        height=250,
        placeholder="Masukkan isi berita di sini..."
    )


# ====================================================
# BUTTON
# ====================================================

classify_button = st.button(
    "🔍 Klasifikasikan Berita",
    type="primary",
    use_container_width=True
)


# ====================================================
# PROSES KLASIFIKASI
# ====================================================

if classify_button:

    # ------------------------------------------------
    # AMBIL TEKS
    # ------------------------------------------------

    if input_type == "URL":

        if not url.strip():

            st.warning(
                "Silakan masukkan URL berita terlebih dahulu."
            )

            st.stop()

        with st.spinner(
            "Mengambil berita dari URL..."
        ):

            article_text = scrape_url(
                url.strip()
            )

        if not article_text:

            st.warning(
                "Teks berita tidak berhasil diambil dari URL."
            )

            st.stop()

    else:

        if not text_input.strip():

            st.warning(
                "Silakan masukkan teks berita terlebih dahulu."
            )

            st.stop()

        article_text = text_input.strip()


    # ------------------------------------------------
    # PREPROCESSING
    # ------------------------------------------------

    with st.spinner(
        "Melakukan preprocessing..."
    ):

        tokens = preprocess_text(
            article_text
        )


    # ------------------------------------------------
    # CEK TOKEN
    # ------------------------------------------------

    if len(tokens) == 0:

        st.warning(
            "Teks berita tidak memiliki kata "
            "yang dapat diproses."
        )

        st.stop()


    # ------------------------------------------------
    # DOMAIN DETECTION
    # ------------------------------------------------

    (
        domain,
        sport_matches,
        finance_matches,
        strong_sport_matches,
        strong_finance_matches
    ) = detect_domain(tokens)


    # =================================================
    # JIKA BUKAN SPORT / FINANCE
    # =================================================

    if domain == "unknown":

        st.warning(
            "Kategori: **Tidak Terdeteksi** ⚠️"
        )

        st.info(
            "Berita ini tidak termasuk kategori "
            "**Sport** atau **Finance**."
        )

        st.write(
            "Naive Bayes tidak digunakan karena "
            "berita tidak terdeteksi sebagai salah satu "
            "dari dua kategori tersebut."
        )

        st.stop()


    # =================================================
    # DOCUMENT VECTOR
    # =================================================

    with st.spinner(
        "Membuat representasi Word2Vec..."
    ):

        doc_vector = document_vector(
            tokens,
            skipgram_model
        )


    # ------------------------------------------------
    # CHECK VECTOR
    # ------------------------------------------------

    if np.all(
        doc_vector == 0
    ):

        st.warning(
            "Tidak ada kata dalam berita yang "
            "ditemukan pada vocabulary Word2Vec."
        )

        st.stop()


    # =================================================
    # SCALING
    # =================================================

    doc_vector_scaled = scaler.transform(
        [doc_vector]
    )


    # =================================================
    # NAIVE BAYES
    # =================================================

    prediction = naive_bayes.predict(
        doc_vector_scaled
    )

    probabilities = naive_bayes.predict_proba(
        doc_vector_scaled
    )[0]


    predicted_class = prediction[0]


    # =================================================
    # HASIL KLASIFIKASI
    # =================================================

    st.divider()

    st.subheader(
        "📊 Hasil Klasifikasi"
    )


    # ------------------------------------------------
    # NORMALISASI NAMA KELAS
    # ------------------------------------------------

    predicted_class_str = str(
        predicted_class
    ).lower()


    if (
        predicted_class_str == "sport"
        or predicted_class_str == "0"
    ):

        result_label = "Sport"

    elif (
        predicted_class_str == "finance"
        or predicted_class_str == "1"
    ):

        result_label = "Finance"

    else:

        result_label = str(
            predicted_class
        )


    # =================================================
    # HASIL UTAMA
    # =================================================

    if result_label == "Sport":

        st.success(
            "🏆 Kategori: **Sport**"
        )

    elif result_label == "Finance":

        st.success(
            "💰 Kategori: **Finance**"
        )

    else:

        st.info(
            f"Kategori: **{result_label}**"
        )


    # =================================================
    # CONFIDENCE
    # =================================================

    confidence = float(
        np.max(probabilities)
    ) * 100


    st.metric(
        "Confidence",
        f"{confidence:.2f}%"
    )


    # =================================================
    # PROBABILITAS
    # =================================================

    st.subheader(
        "📈 Probabilitas Naive Bayes"
    )


    classes = naive_bayes.classes_


    probability_data = {}


    for class_name, probability in zip(
        classes,
        probabilities
    ):

        class_name_str = str(
            class_name
        ).lower()

        if class_name_str == "sport":

            display_name = "Sport"

        elif class_name_str == "finance":

            display_name = "Finance"

        else:

            display_name = str(
                class_name
            )

        probability_data[
            display_name
        ] = f"{probability * 100:.2f}%"


    col1, col2 = st.columns(2)


    with col1:

        if "Sport" in probability_data:

            st.metric(
                "🏆 Sport",
                probability_data["Sport"]
            )


    with col2:

        if "Finance" in probability_data:

            st.metric(
                "💰 Finance",
                probability_data["Finance"]
            )


    # =================================================
    # DETAIL DOMAIN
    # =================================================

    st.divider()

    st.subheader(
        "🔎 Informasi Domain"
    )


    col1, col2 = st.columns(2)


    with col1:

        st.write(
            "**Keyword Sport yang ditemukan:**"
        )

        if sport_matches:

            st.write(
                ", ".join(
                    sorted(sport_matches)
                )
            )

        else:

            st.write(
                "Tidak ada"
            )


        st.write(
            "**Strong Sport Keyword:**"
        )

        if strong_sport_matches:

            st.write(
                ", ".join(
                    sorted(strong_sport_matches)
                )
            )

        else:

            st.write(
                "Tidak ada"
            )


    with col2:

        st.write(
            "**Keyword Finance yang ditemukan:**"
        )

        if finance_matches:

            st.write(
                ", ".join(
                    sorted(finance_matches)
                )
            )

        else:

            st.write(
                "Tidak ada"
            )


        st.write(
            "**Strong Finance Keyword:**"
        )

        if strong_finance_matches:

            st.write(
                ", ".join(
                    sorted(strong_finance_matches)
                )
            )

        else:

            st.write(
                "Tidak ada"
            )


    # =================================================
    # VOCABULARY COVERAGE
    # =================================================

    coverage = calculate_vocabulary_coverage(
        tokens,
        skipgram_model
    )


    st.write(
        f"**Vocabulary Coverage Word2Vec:** "
        f"{coverage:.2f}%"
    )


    # =================================================
    # TEKS BERITA
    # =================================================

    with st.expander(
        "📄 Lihat Teks Berita"
    ):

        st.write(
            article_text
        )


    # =================================================
    # HASIL PREPROCESSING
    # =================================================

    with st.expander(
        "🔤 Lihat Hasil Preprocessing"
    ):

        st.write(
            " ".join(tokens)
        )


# ====================================================
# FOOTER
# ====================================================

st.divider()

st.caption(
    "News Classification — "
    "Skip-Gram Word2Vec + Gaussian Naive Bayes"
)
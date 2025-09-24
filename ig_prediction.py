# streamlit_app.py
import streamlit as st
import pandas as pd
from joblib import load
import numpy as np
import joblib
import re

# --- 1. Daftar kata kunci per kategori ---
kategori_keywords = {
    'Politik': [
        'presiden','menteri','pemilu','partai','kpk','politik','pemerintah','dpr','mk','caleg','capres','pilkada','korupsi', 'kebijakan', 'kementerian',
        'bupati','gubernur','walikota','apbn','apbd','komisi','majelis','legislatif','eksekutif','yudikatif','oposisi','koalisi',
        'kampanye','parlemen','kabinet','perppu','uu','undang','putusan','keppres','perda','permen','rapat','sidang','voting',
        'politikus','konstitusi','hukum','putusan','demokrasi','reformasi','reses','impeachment','hak','interpelasi','anggaran','regulasi'
    ],
    'Lifestyle': [
        'fashion','kuliner','wisata','musik','artis','seleb','gaya','film','hiburan','makeup','olahraga','travel','tren','festival','ekonomi', 'musisi', 'persib', 'liga', 'bandung', 'timnas',
        'makanan','minuman','desain','interior','dekorasi','gadget','smartphone','hobi','fotografi','kecantikan','perawatan','spa','selfcare', 'mobil',
        'liburan','pariwisata','kafe','restoran','hotel','resort','event','pameran','konser','style','aksesoris','pernik','outfit','skincare',
        'ramadan','lebaran','natal','tahunbaru','weekend','shopping','belanja','k-pop','drama'
    ],
    'Human Interest': [
        'kecelakaan','warga','korban','anak','pendidikan','sosial','kemanusiaan','kematian','luka','bencana','banjir','gempa','kebakaran', 'polres', 'polisi', 'israel', 'iran',
        'longsor','topan','tsunami','rawan','penyelamatan','evakuasi','pengungsi','panti','disabilitas','difabel','penderita','pasien','sakit', 'koran', 'dunia',
        'kelaparan','kemiskinan','pengamen','gelandangan','perundungan','bullying','kekerasan','pelecehan','trauma','pernikahan','kelahiran', 'harga',
        'ibu','ayah','keluarga','sahabat','tetangga','masyarakat','lingkungan','relawan','donasi','bantuan','darurat','dampak','psikologis', 'murid'
    ]
}

# --- Fungsi klasifikasi topik ---
def klasifikasi_topik(text):
    text_lower = text.lower()
    skor = {kategori: 0 for kategori in kategori_keywords}

    for kategori, keywords in kategori_keywords.items():
        for kata in keywords:
            if re.search(r'\b' + re.escape(kata) + r'\b', text_lower):
                skor[kategori] += 1

    if all(v == 0 for v in skor.values()):
        return 'Lainnya'
    return max(skor, key=skor.get)

# --- Mapping Waktu Upload untuk tampilan ---
waktu_upload_mapping = {
    "Dini Hari (00:00 - 06:00)": "Dini Hari",
    "Pagi (06:00 - 10:00)": "Pagi",
    "Siang (Jam kerja) (10:00 - 13:00)": "Siang (Jam kerja)",
    "Istirahat Siang (13:00 - 15:00)": "Istirahat Siang",
    "Sore (15:00 - 18:00)": "Sore",
    "Malam (18:00 - 22:00)": "Malam",
    "Larut Malam (22:00 - 00:00)": "Larut Malam"
}

# --- Mapping Kategori Durasi untuk tampilan ---
durasi_mapping = {
    "Non Reels (0 detik)": "Non Reels",
    "Short Reels (<= 60 detik)": "Short Reels",
    "Medium Reels (61 - 300 detik)": "Medium Reels",
    "Long Reels (> 300 detik)": "Long Reels"
}


# 1️⃣ Load model dan mean encoding
best_gb = load("best_gb_model.pkl")
mean_encodings = load("mean_encodings.pkl")

# 2️⃣ Judul
st.title("Prediksi Engagement Rate (%) Instagram")

# 3️⃣ Input user
post_type = st.selectbox("Pilih Post type", ["Reel IG", "Gambar IG", "Carousel IG"])
caption = st.text_area("Tuliskan Caption atau Deskripsi Konten")  # input caption

tipe_hari = st.selectbox("Tentukan Tipe Hari Upload", ["Weekday", "Weekend"])
waktu_upload_input = st.selectbox("Kapan Waktu Upload?", list(waktu_upload_mapping.keys()))
waktu_upload = waktu_upload_mapping[waktu_upload_input]

durasi_input = st.selectbox("Pilih Kategori Durasi Konten", list(durasi_mapping.keys()))
kategori_durasi = durasi_mapping[durasi_input]

# 4️⃣ Buat DataFrame input
# klasifikasi topik dari caption
topic_konten = klasifikasi_topik(caption)

data_baru = pd.DataFrame({
    "Post type": [post_type],
    "Topic Konten": [topic_konten],
    "Tipe Hari": [tipe_hari],
    "Waktu Upload": [waktu_upload],
    "Kategori Durasi": [kategori_durasi]
})

categorical_cols = ['Post type', 'Topic Konten', 'Tipe Hari', 'Waktu Upload', 'Kategori Durasi']

# 5️⃣ Tombol Generate
if st.button("Generate Prediksi"):
    # Apply mean encoding
    for col in categorical_cols:
        data_baru[col] = data_baru[col].map(mean_encodings[col])
        data_baru[col].fillna(data_baru[col].mean(), inplace=True)

      # Prediksi
    prediksi_engagement = best_gb.predict(data_baru)

    # Tambahkan rentang kemungkinan nilai asli
    rmse_test = 2.21 # RMSE dari evaluasi model
    lower_bound = max(0, prediksi_engagement - rmse_test)
    upper_bound = prediksi_engagement + rmse_test


    st.success(f"Prediksi Engagement Rate (%): {prediksi_engagement[0]:.2f}")
    st.info(f"Rentang kemungkinan nilai asli: {lower_bound[0]:.2f}% - {upper_bound[0]:.2f}%")

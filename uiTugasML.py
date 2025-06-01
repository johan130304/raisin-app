import streamlit as st
import numpy as np
import pickle
import joblib
from PIL import Image

# Load model dan parameter normalisasi
model = joblib.load('stacking_model.pkl')
mean = np.load('mean.npy')
std_dev = np.load('std_dev.npy')

# Menambahkan gambar yang sudah ada dari file
image = Image.open('raisin.jpg')  # Ganti dengan path gambar raisin.jpg yang sudah ada

# Menampilkan judul di atas gambar menggunakan HTML
st.markdown("<h1 style='text-align: center; color: white;'>Klasifikasi Raisin</h1>", unsafe_allow_html=True)

# Menampilkan gambar dengan parameter 'use_container_width=True'
st.image(image, use_container_width=True)

# Penjelasan aplikasi di atas input data
st.markdown("""
    ### Penjelasan:
    - **Area**: Luas dari raisin (anggur kering).
    - **MajorAxisLength**: Panjang sumbu utama dari raisin.
    - **MinorAxisLength**: Panjang sumbu minor dari raisin.
    - **Eccentricity**: Mengukur seberapa jauh bentuknya dari lingkaran sempurna.
    - **ConvexArea**: Luas area konveks (batas luar) dari raisin.
    - **Extent**: Rasio luas objek ke area bounding box.
    - **Perimeter**: Keliling dari objek raisin.

    Data ini digunakan oleh model **Stacking Classifier** untuk mengklasifikasikan raisin ke dalam kategori yang tepat.
    
    Silakan masukkan nilai fitur untuk melakukan prediksi!
""")

# Input fitur (ganti label sesuai fitur di dataset Raisin)
st.subheader("Masukkan Nilai Fitur Raisin")

feature_names = ['Area', 'MajorAxisLength', 'MinorAxisLength', 'Eccentricity', 'ConvexArea', 'Extent', 'Perimeter']
user_input = []
for feature in feature_names:
    value = st.number_input(f"Masukkan nilai untuk {feature}", format="%.5f", min_value=0.0)
    user_input.append(value)

if st.button("Prediksi"):
    # Konversi ke array dan normalisasi
    input_array = np.array(user_input).reshape(1, -1)
    normalized_input = (input_array - mean) / std_dev

    # Prediksi
    prediction = model.predict(normalized_input)[0]
    
    # Output prediksi
    st.success(f"Model memprediksi kelas: **{prediction}**")

    # Visualisasi hasil
    if prediction == 1:
        st.markdown("Raisin ini termasuk dalam **kategori Besni**.")
    else:
        st.markdown("Raisin ini termasuk dalam **kategori Kesimen**.")

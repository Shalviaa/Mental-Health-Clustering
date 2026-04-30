"""
================================================================================
MENTAL HEALTH CLUSTERING - STREAMLIT DEPLOYMENT APP
Metodologi: CRISP-DM (Deployment Phase)
Model: Word2Vec + PCA100 + K-Means Clustering
================================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
import string
from collections import Counter

# Load model and data
def load_models():
    """Load all necessary models and data"""
    try:
        with open('w2v_model_tuned_pca100.pkl', 'rb') as f:
            w2v_model = pickle.load(f)
        with open('pca_model.pkl', 'rb') as f:
            pca_model = pickle.load(f)
        with open('kmeans_model.pkl', 'rb') as f:
            kmeans_model = pickle.load(f)
        with open('cluster_info.pkl', 'rb') as f:
            cluster_info = pickle.load(f)
        return w2v_model, pca_model, kmeans_model, cluster_info
    except:
        return None, None, None, None

# Text preprocessing
def preprocess_text(text):
    """Basic text preprocessing"""
    # Lowercase
    text = text.lower()
    # Remove URLs, mentions, hashtags
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+|#\w+', '', text)
    # Remove numbers and punctuation
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenize
    tokens = text.split()
    # Remove empty tokens
    tokens = [t for t in tokens if len(t) > 1]
    return tokens

# Create document vector
def create_doc_vector(tokens, w2v_model, vector_size=300):
    """Create document vector from Word2Vec"""
    vectors = []
    for word in tokens:
        if word in w2v_model.wv:
            vectors.append(w2v_model.wv[word])
    if vectors:
        return np.mean(vectors, axis=0)
    return np.zeros(vector_size)

# Predict cluster
def predict_cluster(text, w2v_model, pca_model, kmeans_model):
    """Predict cluster for input text"""
    # Preprocess
    tokens = preprocess_text(text)
    if len(tokens) < 3:
        return None, None, tokens, "Teks terlalu pendek (minimum 3 kata)"
    
    # Create vector
    doc_vec = create_doc_vector(tokens, w2v_model, 300)
    
    # Apply PCA
    doc_vec_pca = pca_model.transform([doc_vec])
    
    # Predict
    cluster = kmeans_model.predict(doc_vec_pca)[0]
    
    # Calculate distances to all centroids for confidence
    distances = kmeans_model.transform(doc_vec_pca)[0]
    # Convert to probabilities (softmax-like)
    exp_distances = np.exp(-distances)
    probabilities = exp_distances / exp_distances.sum()
    confidence = probabilities[cluster] * 100
    
    return cluster, confidence, tokens, None

# Get cluster bigrams
def get_cluster_bigrams(cluster_id, cluster_info):
    """Get top bigrams for cluster"""
    return cluster_info.get(cluster_id, {}).get('bigrams', [])

# Main app
st.set_page_config(
    page_title="Mental Health Clustering",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .cluster-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .cluster-0 {
        background-color: #ffcccc;
        border-left: 5px solid #cc0000;
    }
    .cluster-1 {
        background-color: #cce5ff;
        border-left: 5px solid #0066cc;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
    .bigram-tag {
        display: inline-block;
        background-color: #e9ecef;
        padding: 0.3rem 0.8rem;
        margin: 0.2rem;
        border-radius: 15px;
        font-size: 0.9rem;
    }
    .footer {
        text-align: center;
        color: #666;
        font-size: 0.8rem;
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #ddd;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="main-header">🧠 Sistem Deteksi Kesehatan Mental</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Clustering Curhat Online dengan Word2Vec + PCA + K-Means</p>', unsafe_allow_html=True)

# Load models
models = load_models()
models_loaded = all(m is not None for m in models)

if not models_loaded:
    st.warning("⚠️ Model belum tersedia. Menggunakan mode demonstrasi dengan data simulasi.")
    # Demo mode with sample data
    cluster_info_demo = {
        0: {
            'name': 'Depresi & Gejala Fisik',
            'description': (
                'Cluster ini menunjukkan indikasi depresi dengan dominasi emosi negatif dan gejala fisik. '
                'Pola bigram menampilkan ciri-ciri distress emosional (putus asa, keluh kesah) yang disertai '
                'pelampiasan emosi (caci maki) dan manifestasi somatik (sesak nafas, keringat dingin). '
                'Individu dalam cluster ini cenderung mengalami gejala depresi dengan komponen fisik yang signifikan.'
            ),
            'bigrams': [
                ('putus', 'asa', 49),
                ('keluh', 'kesah', 35),
                ('caci', 'maki', 28),
                ('sesak', 'nafas', 27),
                ('keringat', 'dingin', 26)
            ],
            'recommendation': (
                '1. Segera konsultasi ke psikolog/psikiater untuk evaluasi lebih lanjut.\n'
                '2. Lakukan self-care: tidur cukup 7-8 jam, makan teratur, olahraga ringan.\n'
                '3. Jangan ragu mencari support system (keluarga, teman, komunitas).\n'
                '4. Jika ada pikiran untuk menyakiti diri, hubungi 119 ext 8 (hotline kesehatan mental).'
            )
        },
        1: {
            'name': 'Kecemasan & Kesadaran Mental',
            'description': (
                'Cluster ini merepresentasikan kecemasan yang masih disertai kesadaran akan pentingnya kesehatan mental. '
                'Pola bigram menunjukkan komorbiditas antara kecemasan (cemas lebih, cemas takut) dan kesadaran positif '
                '(sehat mental) serta gejala somatik (sesak nafas). Individu aware akan kondisinya namun masih '
                'mengalami anxiety yang mengganggu aktivitas sehari-hari.'
            ),
            'bigrams': [
                ('putus', 'asa', 80),
                ('sehat', 'mental', 71),
                ('cemas', 'lebih', 68),
                ('sesak', 'nafas', 58),
                ('cemas', 'takut', 50)
            ],
            'recommendation': (
                '1. Praktikkan teknik relaksasi: deep breathing (4-7-8), mindfulness meditation.\n'
                '2. Gunakan grounding techniques: 5-4-3-2-1 senses technique saat anxiety menyerang.\n'
                '3. Batasi konsumsi kafein dan berita negatif.\n'
                '4. Jika gejala persisten >2 minggu, konsultasi ke profesional kesehatan mental.'
            )
        }
    }
else:
    w2v_model, pca_model, kmeans_model, cluster_info = models

# Main layout
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 📝 Input Teks Curhat")
    
    # Example buttons
    st.markdown("**Contoh Input:**")
    example_cols = st.columns(3)
    example_texts = [
        "Saya merasa putus asa, sulit tidur, badan lemas, dan sering sesak napas",
        "Saya cemas terus menerus tentang masa depan, takut gagal, dan sering overthinking",
        "Saya merasa diri ini tidak berharga, selalu dikeluhan, dan ingin menangis"
    ]
    
    selected_example = None
    for i, (col, text) in enumerate(zip(example_cols, example_texts)):
        if col.button(f"Contoh {i+1}", key=f"ex_{i}"):
            selected_example = text
    
    # Text input
    if selected_example:
        user_input = st.text_area("Tulis curhat Anda di sini:", value=selected_example, height=200)
    else:
        user_input = st.text_area("Tulis curhat Anda di sini:", placeholder="Minimal 10 kata...", height=200)
    
    # Action buttons
    btn_col1, btn_col2 = st.columns(2)
    analyze_clicked = btn_col1.button("🔍 ANALISIS", use_container_width=True)
    clear_clicked = btn_col2.button("🗑️ HAPUS", use_container_width=True)
    
    if clear_clicked:
        st.session_state.clear()
        st.experimental_rerun()
    
    # Info box
    st.info("ℹ️ **Catatan:** Sistem ini menggunakan model clustering untuk mengelompokkan teks berdasarkan pola kemiripan. Ini BUKAN diagnosis medis.")

with col2:
    if analyze_clicked and user_input:
        if len(user_input.split()) < 3:
            st.error("⚠️ Teks terlalu pendek. Minimal 3 kata.")
        else:
            with st.spinner("Menganalisis..."):
                # Simulate processing delay
                import time
                time.sleep(1)
                
                # Demo prediction (in real app, use models)
                if not models_loaded:
                    # Simple keyword-based demo
                    anxiety_keywords = ['cemas', 'takut', 'khawatir', 'overthinking', 'panik', 'anxiety']
                    depression_keywords = ['putus asa', 'sedih', 'lemes', 'tidak berharga', 'gagal', 'menangis', 'depresi']
                    
                    text_lower = user_input.lower()
                    anxiety_score = sum(1 for k in anxiety_keywords if k in text_lower)
                    depression_score = sum(1 for k in depression_keywords if k in text_lower)
                    
                    if anxiety_score > depression_score:
                        cluster_id = 1
                        confidence = min(50 + anxiety_score * 10, 95)
                    else:
                        cluster_id = 0
                        confidence = min(50 + depression_score * 10, 95)
                    
                    cluster_data = cluster_info_demo[cluster_id]
                else:
                    cluster_id, confidence, tokens, error = predict_cluster(user_input, w2v_model, pca_model, kmeans_model)
                    if error:
                        st.error(f"⚠️ {error}")
                        cluster_data = None
                    else:
                        cluster_data = cluster_info.get(cluster_id, {})
                
                if cluster_data:
                    # Result display
                    st.markdown("### 📊 Hasil Analisis")
                    
                    # Cluster box
                    cluster_class = f"cluster-{cluster_id}"
                    cluster_name = cluster_data.get('name', f'Cluster {cluster_id}')
                    
                    st.markdown(f'''
                    <div class="cluster-box {cluster_class}">
                        <h4>Cluster Terdeteksi: {cluster_name}</h4>
                        <p><strong>Tingkat Kepercayaan: {confidence:.1f}%</strong></p>
                    </div>
                    ''', unsafe_allow_html=True)
                    
                    # Metrics
                    metric_cols = st.columns(3)
                    metric_cols[0].metric("Jumlah Kata", len(user_input.split()))
                    metric_cols[1].metric("Cluster ID", f"#{cluster_id}")
                    metric_cols[2].metric("Confidence", f"{confidence:.1f}%")
                    
                    # Tabs for detailed info
                    tab1, tab2, tab3 = st.tabs(["📌 Gejala Terdeteksi", "📖 Penjelasan", "💡 Rekomendasi"])
                    
                    with tab1:
                        st.markdown("**Top Bigram dalam Cluster ini:**")
                        bigrams = cluster_data.get('bigrams', [])
                        for bigram in bigrams[:5]:
                            # Format: (word1, word2, frequency)
                            if len(bigram) == 3:
                                w1, w2, freq = bigram
                                st.markdown(f'<span class="bigram-tag">{w1} {w2} ({freq})</span>', unsafe_allow_html=True)
                            else:
                                st.markdown(f'<span class="bigram-tag">{str(bigram)}</span>', unsafe_allow_html=True)
                    
                    with tab2:
                        st.markdown(cluster_data.get('description', 'Tidak ada deskripsi'))
                    
                    with tab3:
                        st.success(cluster_data.get('recommendation', 'Konsultasi ke profesional kesehatan mental'))
    else:
        st.markdown("### 📊 Hasil Analisis")
        st.info("👈 Masukkan teks curhat dan klik 'ANALISIS' untuk melihat hasil clustering.")

# Footer
st.markdown('---')
footer_cols = st.columns(3)
footer_cols[0].metric("Dataset", "5,694 dokumen")
footer_cols[1].metric("Model", "Word2Vec + PCA100")
footer_cols[2].metric("Cluster Optimal", "K=2")

st.markdown('<p class="footer">© 2026 - Skripsi Clustering Kesehatan Mental | Metodologi CRISP-DM</p>', unsafe_allow_html=True)

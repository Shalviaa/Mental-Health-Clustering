# 🚀 DEPLOYMENT GUIDE - MENTAL HEALTH CLUSTERING APP

**Metodologi:** CRISP-DM (Phase 6: Deployment)  
**Framework:** Streamlit  
**Model:** Word2Vec + PCA100 + K-Means Clustering

---

## 📁 Struktur Folder Deployment

```
Deployment/
├── app.py                    # Aplikasi Streamlit utama
├── requirements.txt          # Dependencies
├── README_DEPLOYMENT.md      # Panduan ini
├── w2v_model_tuned_pca100.pkl   # Model Word2Vec (export dari notebook)
├── pca_model.pkl             # Model PCA (export dari notebook)
├── kmeans_model.pkl          # Model K-Means (export dari notebook)
└── cluster_info.pkl          # Informasi cluster (bigrams, deskripsi)
```

---

## 🎯 Langkah Deployment

### **STEP 1: Export Model dari Notebook**

Tambahkan cell di akhir notebook `modelling_w2v_lemma_pca100_keepword_tuned.ipynb`:

```python
import pickle

# Save Word2Vec model
w2v_model_final.save('w2v_model_tuned_pca100.pkl')

# Save PCA model
with open('pca_model.pkl', 'wb') as f:
    pickle.dump(pca_final, f)

# Save KMeans model
with open('kmeans_model.pkl', 'wb') as f:
    pickle.dump(kmeans_final, f)

# Save cluster info
cluster_info = {
    0: {
        'name': 'Depresi & Gejala Fisik',
        'description': 'Cluster ini menunjukkan indikasi depresi dengan gejala fisik...',
        'bigrams': [('putus asa', 49), ('keluh kesah', 35), ('caci maki', 28), ('sesak nafas', 27), ('keringat dingin', 26)],
        'recommendation': 'Segera konsultasi ke psikolog/psikiater...'
    },
    1: {
        'name': 'Kecemasan & Kesadaran Mental',
        'description': 'Cluster ini menunjukkan kecemasan yang masih disertai kesadaran...',
        'bigrams': [('putus asa', 80), ('sehat mental', 71), ('cemas lebih', 68), ('sesak nafas', 58), ('cemas takut', 50)],
        'recommendation': 'Praktikkan teknik relaksasi: deep breathing...'
    }
}

with open('cluster_info.pkl', 'wb') as f:
    pickle.dump(cluster_info, f)

print("✅ Model dan info cluster berhasil disimpan!")
```

Copy file `.pkl` yang dihasilkan ke folder `Deployment/`.

---

### **STEP 2: Setup Environment**

```bash
# Navigate to deployment folder
cd c:\KULIAH\Skripshiiii\TA_SHALVIA\Deployment

# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

---

### **STEP 3: Test Lokal**

```bash
# Jalankan aplikasi
streamlit run app.py

# Buka browser: http://localhost:8501
```

---

### **STEP 4: Deploy ke Streamlit Cloud (FREE)**

#### **Opsi A: Deploy via GitHub (Recommended)**

1. **Buat repository GitHub:**
   ```bash
   # Inisialisasi git di folder Deployment
   git init
   git add .
   git commit -m "Initial deployment"
   
   # Push ke GitHub (buat repo baru di github.com)
   git remote add origin https://github.com/username/mental-health-clustering.git
   git push -u origin main
   ```

2. **Deploy di Streamlit Cloud:**
   - Buka https://share.streamlit.io
   - Sign in dengan GitHub
   - Click "New app"
   - Pilih repository Anda
   - File path: `app.py`
   - Click "Deploy"
   - Done! 🎉

#### **Opsi B: Deploy Lokal dengan ngrok (Testing)**

```bash
# Install ngrok
pip install pyngrok

# Jalankan streamlit dengan tunnel
streamlit run app.py &
ngrok http 8501
```

---

## 📊 Fitur Aplikasi Web

```
┌─────────────────────────────────────────────────────────────┐
│           🧠 SISTEM DETEKSI KESEHATAN MENTAL               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  INPUT (Kiri)              │  OUTPUT (Kanan)               │
│  ──────────────────────────┼────────────────────────────────│
│  • Contoh input buttons    │  • Cluster terdeteksi          │
│  • Text area input        │  • Confidence score            │
│  • Analyze button         │  • Gejala terdeteksi           │
│  • Clear button           │  • Penjelasan cluster          │
│                           │  • Rekomendasi aksi            │
│                           │                               │
│  +---------------------+  │  +-------------------------+ │
│  | ⚠️ Disclaimer:     |  │  | 📊 Metrics:             | │
│  | Bukan diagnosis    |  │  | • Dataset: 5,694        | │
│  | medis             |  │  | • Model: W2V+PCA+KMeans | │
│  +---------------------+  │  | • K Optimal: 2          | │
│                           │  +-------------------------+ │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔧 Troubleshooting

| Error | Solusi |
|-------|--------|
| `ModuleNotFoundError` | `pip install -r requirements.txt` |
| Model not found | Pastikan file `.pkl` di folder yang sama dengan `app.py` |
| Out of memory | Kurangi `vector_size` di Word2Vec atau gunakan PCA lebih agresif |
| Deploy failed | Check log di Streamlit Cloud, pastikan semua dependencies terdaftar |

---

## 📚 Dokumentasi untuk Skripsi (Bab Deployment)

### **Bab 4 / Bab 5: Deployment**

> **4.X Deployment Sistem**
>
> Pada tahap deployment, model clustering yang telah dibangun diimplementasikan dalam bentuk aplikasi web berbasis Streamlit. Streamlit dipilih karena kemudahan deployment, native support untuk machine learning visualization, dan tidak memerlukan pengetahuan frontend development yang mendalam.
>
> **Arsitektur Deployment:**
> - **Backend:** Python dengan library scikit-learn untuk loading model K-Means dan PCA, serta gensim untuk Word2Vec embedding
> - **Frontend:** Streamlit yang menyediakan UI components built-in
> - **Hosting:** Streamlit Cloud (gratis) dengan continuous deployment dari GitHub
>
> **Alur Kerja Aplikasi:**
> 1. User memasukkan teks curhat melalui text area
> 2. Sistem melakukan preprocessing (tokenisasi, lowercase, cleaning)
> 3. Text dikonversi ke vector menggunakan Word2Vec model
> 4. Vector direduksi dimensinya dengan PCA (100 komponen)
> 5. K-Means model memprediksi cluster (0: Depresi, 1: Kecemasan)
> 6. Sistem menampilkan hasil dengan confidence score dan rekomendasi
>
> **URL Aplikasi:** `https://[username]-mental-health-app.streamlit.app`

---

## ✅ Checklist Deployment

- [ ] Model Word2Vec di-export ke `.pkl`
- [ ] Model PCA di-export ke `.pkl`
- [ ] Model K-Means di-export ke `.pkl`
- [ ] Cluster info (bigrams, deskripsi) di-export
- [ ] `requirements.txt` lengkap dengan versi
- [ ] Test lokal berhasil
- [ ] Upload ke GitHub
- [ ] Deploy ke Streamlit Cloud
- [ ] URL aplikasi tercatat di skripsi

---

## 🎓 Catatan Penting

⚠️ **Disclaimer untuk Aplikasi:**
- Sistem ini **BUKAN alat diagnosis medis**
- Hasil clustering berdasarkan pola statistik, bukan diagnosis klinis
- Selalu sarankan konsultasi ke profesional kesehatan mental
- Tujuannya adalah screening awal dan edukasi

---

**Dibuat:** April 2026  
**Framework:** Streamlit 1.28.0  
**Model:** Word2Vec + PCA100 + K-Means (K=2 Optimal)

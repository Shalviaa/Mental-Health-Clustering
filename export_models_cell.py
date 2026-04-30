"""
================================================================================
CELL UNTUK EXPORT MODEL DARI NOTEBOOK
Tambahkan cell ini ke akhir notebook: modelling_w2v_lemma_pca100_keepword_tuned.ipynb
================================================================================

Setelah di-run, copy file .pkl yang dihasilkan ke folder Deployment/
================================================================================
"""

import pickle
import pandas as pd
from collections import Counter
from nltk import bigrams

print("="*70)
print("EXPORT MODEL UNTUK DEPLOYMENT")
print("="*70)

# 1. Export Word2Vec model
w2v_model_final.save('w2v_model_tuned_pca100.pkl')
print("✅ Word2Vec model saved: w2v_model_tuned_pca100.pkl")

# 2. Export PCA model
with open('pca_model.pkl', 'wb') as f:
    pickle.dump(pca_final, f)
print("✅ PCA model saved: pca_model.pkl")

# 3. Export KMeans model
with open('kmeans_model.pkl', 'wb') as f:
    pickle.dump(kmeans_final, f)
print("✅ KMeans model saved: kmeans_model.pkl")

# 4. Export Cluster Info (dari hasil analisis)
cluster_info = {
    0: {
        'name': 'Depresi & Gejala Fisik',
        'description': (
            'Cluster ini menunjukkan indikasi depresi dengan dominasi emosi negatif '
            'dan gejala fisik. Pola bigram menampilkan ciri-ciri distress emosional '
            '(putus asa, keluh kesah) yang disertai pelampiasan emosi (caci maki) '
            'dan manifestasi somatik (sesak nafas, keringat dingin). '
            'Individu dalam cluster ini cenderung mengalami gejala depresi dengan '
            'komponen fisik yang signifikan.'
        ),
        'bigrams': [
            ('putus', 'asa', 49),
            ('keluh', 'kesah', 35),
            ('caci', 'maki', 28),
            ('sesak', 'nafas', 27),
            ('keringat', 'dingin', 26)
        ],
        'recommendation': (
            '1. Segera konsultasi ke psikolog atau psikiater untuk evaluasi lebih lanjut.\n'
            '2. Lakukan self-care: tidur cukup 7-8 jam, makan teratur, olahraga ringan.\n'
            '3. Jangan ragu mencari support system (keluarga, teman, komunitas).\n'
            '4. Jika ada pikiran untuk menyakiti diri, hubungi 119 ext 8 (hotline kesehatan mental).'
        )
    },
    1: {
        'name': 'Kecemasan & Kesadaran Mental',
        'description': (
            'Cluster ini merepresentasikan kecemasan yang masih disertai kesadaran '
            'akan pentingnya kesehatan mental. Pola bigram menunjukkan komorbiditas '
            'antara kecemasan (cemas lebih, cemas takut) dan kesadaran positif '
            '(sehat mental) serta gejala somatik (sesak nafas). '
            'Individu dalam cluster ini aware akan kondisinya namun masih '
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

with open('cluster_info.pkl', 'wb') as f:
    pickle.dump(cluster_info, f)
print("✅ Cluster info saved: cluster_info.pkl")

print("\n" + "="*70)
print("SUMMARY FILE YANG DIHASILKAN:")
print("="*70)
print("1. w2v_model_tuned_pca100.pkl - Model Word2Vec (300 dimensi)")
print("2. pca_model.pkl - Model PCA (100 komponen)")
print("3. kmeans_model.pkl - Model K-Means (K=2 clusters)")
print("4. cluster_info.pkl - Informasi cluster (bigrams, deskripsi)")
print("\nCOPY KEEMPAT FILE INI KE FOLDER Deployment/")
print("="*70)

# ================================================================================
# TAMBAHKAN CELL INI KE AKHIR NOTEBOOK
# modelling_w2v_lemma_pca100_keepword_tuned.ipynb
# ================================================================================

import pickle
import pandas as pd
from collections import Counter
from nltk import bigrams

print("="*70)
print("EXPORT MODEL UNTUK DEPLOYMENT")
print("="*70)

# 1. Export PCA model (sudah ada di variabel pca_final)
with open('pca_model.pkl', 'wb') as f:
    pickle.dump(pca_final, f)
print("✅ PCA model (100 components) saved: pca_model.pkl")

# 2. Export KMeans model (sudah ada di variabel kmeans_final)
with open('kmeans_model.pkl', 'wb') as f:
    pickle.dump(kmeans_final, f)
print("✅ KMeans model (K=2) saved: kmeans_model.pkl")

# 3. Generate dan export cluster info dari data
print("\nGenerating cluster info from data...")

cluster_info = {}
for cluster_id in range(optimal_k):
    # Get text for this cluster
    cluster_text = ' '.join(df[df['cluster']==cluster_id]['lemmas'])
    tokens = cluster_text.split()
    
    # Get top bigrams
    bg = Counter(bigrams(tokens)).most_common(5)
    
    # Format bigrams as (word1, word2, frequency)
    top_bigrams = [(w1, w2, freq) for (w1, w2), freq in bg]
    
    # Determine cluster name and description based on bigrams
    if cluster_id == 0:
        name = 'Depresi & Gejala Fisik'
        description = (
            'Cluster ini menunjukkan indikasi depresi dengan dominasi emosi negatif '
            'dan gejala fisik. Pola bigram menampilkan ciri-ciri distress emosional '
            '(putus asa, keluh kesah) yang disertai pelampiasan emosi (caci maki) '
            'dan manifestasi somatik (sesak nafas, keringat dingin).'
        )
        recommendation = (
            '1. Segera konsultasi ke psikolog/psikiater.\n'
            '2. Lakukan self-care: tidur cukup, olahraga ringan.\n'
            '3. Cari support system (keluarga, teman).\n'
            '4. Hotline: 119 ext 8 (kesehatan mental).'
        )
    else:
        name = 'Kecemasan & Kesadaran Mental'
        description = (
            'Cluster ini merepresentasikan kecemasan yang masih disertai kesadaran '
            'akan kesehatan mental. Pola bigram menunjukkan komorbiditas antara '
            'kecemasan dan kesadaran positif serta gejala somatik.'
        )
        recommendation = (
            '1. Praktikkan deep breathing dan mindfulness.\n'
            '2. Gunakan grounding techniques saat anxiety.\n'
            '3. Batasi kafein dan berita negatif.\n'
            '4. Konsultasi jika gejala >2 minggu.'
        )
    
    cluster_info[cluster_id] = {
        'name': name,
        'description': description,
        'bigrams': top_bigrams,
        'recommendation': recommendation
    }
    
    print(f"\nCluster {cluster_id}: {name}")
    print(f"  Top bigrams: {top_bigrams}")

# Save cluster_info
with open('cluster_info.pkl', 'wb') as f:
    pickle.dump(cluster_info, f)
print("\n✅ Cluster info saved: cluster_info.pkl")

print("\n" + "="*70)
print("SUMMARY FILE UNTUK DEPLOYMENT:")
print("="*70)
print("1. pca_model.pkl - Model PCA (100 components)")
print("2. kmeans_model.pkl - Model K-Means (K=2)")
print("3. cluster_info.pkl - Info cluster (bigrams, deskripsi)")
print("\nCopy ketiga file ini ke folder Deployment/")
print("W2V model sudah ada: w2v_model_tuned_pca100.pkl")
print("="*70)

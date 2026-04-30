"""
================================================================================
EXTRACT CLUSTER INFO FROM NOTEBOOK RESULTS
Script untuk mengekstrak informasi cluster dari hasil notebook
================================================================================

Data cluster_info meliputi:
- name: Nama cluster berdasarkan tema bigram
- description: Interpretasi detail dari pola bigram
- bigrams: Top 5 bigram dengan frequency
- recommendation: Saran tindakan berdasarkan tema cluster

Sumber data: README_PCA100_TUNED.txt / notebook results
================================================================================
"""

import pickle

# Data cluster diambil dari hasil analisis di README_PCA100_TUNED.txt
# K=2 (Optimal berdasarkan metrik)

CLUSTER_INFO_K2 = {
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
            ('putus asa', 49),
            ('keluh kesah', 35),
            ('caci maki', 28),
            ('sesak nafas', 27),
            ('keringat dingin', 26)
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
            ('putus asa', 80),
            ('sehat mental', 71),
            ('cemas lebih', 68),
            ('sesak nafas', 58),
            ('cemas takut', 50)
        ],
        'recommendation': (
            '1. Praktikkan teknik relaksasi: deep breathing (4-7-8), mindfulness meditation.\n'
            '2. Gunakan grounding techniques: 5-4-3-2-1 senses technique saat anxiety menyerang.\n'
            '3. Batasi konsumsi kafein dan berita negatif.\n'
            '4. Jika gejala persisten >2 minggu, konsultasi ke profesional kesehatan mental.\n'
            '5. Pertahankan kesadaran positif tentang kesehatan mental.'
        )
    }
}

# Untuk K=3 (alternatif untuk analisis tematik)
CLUSTER_INFO_K3 = {
    0: {
        'name': 'Distress Emosional Berat',
        'description': (
            'Depresi dengan pelampiasan emosi dan indikasi ketidakstabilan emosi. '
            'Muncul bigram episode mania yang mengindikasikan adanya komponen '
            'mood disorder yang lebih kompleks.'
        ),
        'bigrams': [
            ('putus asa', 49),
            ('keluh kesah', 35),
            ('caci maki', 28),
            ('episode mania', 25),
            ('picu episode', 22)
        ],
        'recommendation': 'Konsultasi segera ke psikiater untuk evaluasi mood disorder.'
    },
    1: {
        'name': 'Krisis Emosional & Kebingungan',
        'description': (
            'Krisis emosional dengan overthinking, kehilangan arah, dan penarikan diri. '
            'Masih ada refleksi positif terhadap kondisi mental.'
        ),
        'bigrams': [
            ('putus asa', 60),
            ('sehat mental', 55),
            ('hilang arah', 45),
            ('takut hilang', 40),
            ('pilih diam', 35)
        ],
        'recommendation': 'Cari support system, pertimbangkan therapy/counseling.'
    },
    2: {
        'name': 'Anxiety Somatik',
        'description': (
            'Kecemasan yang berdampak pada tubuh dengan gejala somatik jelas. '
            'Manifestasi fisik seperti sesak napas dan jantung berdebar dominan.'
        ),
        'bigrams': [
            ('cemas lebih', 68),
            ('sesak nafas', 58),
            ('cemas takut', 50),
            ('badan lemas', 45),
            ('jantung debar', 40)
        ],
        'recommendation': 'Latih teknik pernapasan, olahraga ringan, meditasi.'
    }
}

def save_cluster_info(k=2):
    """Save cluster info to pickle file"""
    if k == 2:
        cluster_info = CLUSTER_INFO_K2
    elif k == 3:
        cluster_info = CLUSTER_INFO_K3
    else:
        raise ValueError("Only K=2 and K=3 are supported")
    
    output_file = f'cluster_info_k{k}.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump(cluster_info, f)
    
    print(f"✅ Cluster info (K={k}) saved to: {output_file}")
    print(f"\n📊 Summary:")
    for cluster_id, info in cluster_info.items():
        print(f"  Cluster {cluster_id}: {info['name']}")
        print(f"    - Top bigram: {info['bigrams'][0][0]} {info['bigrams'][0][1]} ({info['bigrams'][0][2]})")
    
    return output_file

if __name__ == "__main__":
    # Save K=2 (recommended for deployment)
    save_cluster_info(k=2)
    
    # Uncomment untuk K=3:
    # save_cluster_info(k=3)

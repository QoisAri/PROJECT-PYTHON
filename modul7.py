from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# Membuat data customer dalam bentuk dataframe
data = pd.DataFrame({
    'Usia': [30, 20, 35, 25, 40, 30, 45, 35, 50, 40],
    'Pendapatan': [2000, 1500, 2500, 1800, 3000, 2200, 3500, 2800, 4000, 3200],
    'Jumlah Pembelian': [5, 3, 7, 4, 9, 6, 11, 8, 13, 10],
    'Jenis Kelamin': ['Laki-laki', 'Perempuan', 'Laki-laki', 'Perempuan', 'Laki-laki', 'Perempuan', 'Laki-laki', 'Perempuan', 'Laki-laki', 'Perempuan'],
    'Status Pernikahan': ['Belum Menikah', 'Belum Menikah', 'Menikah', 'Belum Menikah', 'Menikah', 'Menikah', 'Belum Menikah', 'Menikah', 'Menikah', 'Menikah']
})

# Memilih kolom yang akan dinormalisasi
kolom = ['Usia', 'Pendapatan', 'Jumlah Pembelian']

# Menggunakan metode MinMaxScaler untuk normalisasi data
scaler = MinMaxScaler()
data[kolom] = scaler.fit_transform(data[kolom])
# Mengubah nilai 'Laki-laki' menjadi 1 dan 'Perempuan' menjadi 0 pada atribut jenis kelamin
data['Jenis Kelamin'] = data['Jenis Kelamin'].replace({'Laki-laki': 1, 'Perempuan': 0})

# Mengubah nilai 'Menikah' menjadi 1 dan 'Belum Menikah' menjadi 0 pada atribut Status Pernikahan
data['Status Pernikahan'] = data['Status Pernikahan'].replace({'Belum Menikah': 0, 'Menikah': 1})

# Menampilkan hasil normalisasi data
print(data)

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Memilih kolom yang akan dijadikan fitur untuk pengelompokan
fitur = ['Usia', 'Pendapatan', 'Jumlah Pembelian']
X = data[fitur]

# Melakukan analisis elbow untuk menentukan nilai k terbaik
wcss = []
for i in range(1, 7):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
# Menampilkan visualisasi elbow
plt.plot(range(1, 7), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters (k)')
plt.ylabel('WCSS')
plt.show()

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

# Melakukan pengelompokan data customer menggunakan algoritma k-means clustering
jumlah_cluster = 2
kmeans = KMeans(n_clusters=jumlah_cluster, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(X)

# Melakukan reduksi dimensi menggunakan PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
# Melakukan reduksi dimensi menggunakan PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Menampilkan visualisasi clustering dalam scatter plot
for i in range(jumlah_cluster):
    plt.scatter(X_pca[y_kmeans==i, 0], X_pca[y_kmeans==i, 1],
                s=100, c=np.random.rand(3,), label='Cluster {}'.format(i))

plt.title('Customer Segmentation')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.show()
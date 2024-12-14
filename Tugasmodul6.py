import numpy as np
import matplotlib.pyplot as plt

# Contoh data buah
data = np.array([
    [150, 6, 'Apel'],
    [175, 7, 'Apel'],
    [200, 8, 'Apel'],
    [225, 9, 'Apel'],
    [250, 10, 'Apel'],
    [275, 11, 'Apel'],
    [140, 5, 'Jeruk'],
    [165, 6, 'Jeruk'],
    [190, 7, 'Jeruk'],
    [215, 8, 'Jeruk'],
    [240, 9, 'Jeruk'],
    [265, 10, 'Jeruk']
])

# Buah yang akan diklasifikasikan
buah = np.array([220, 8])

# Hitung jarak antara buah yang akan diklasifikasikan dengan setiap data
jarak = np.sqrt(np.sum((data[:, :2].astype(int) - buah) ** 2, axis=1))

# Tentukan nilai k dan pilih k titik data terdekat
k = 3
idx = np.argsort(jarak)[:k]
k_titik_data = data[idx]

# Tentukan kelas mayoritas dari k titik data terdekat 
kelas = np.unique(k_titik_data[:, 2])
jumlah_kelas = np.array([len(k_titik_data[k_titik_data[:, 2] == kl]) for kl in kelas])
kelas_mayoritas = kelas[np.argmax(jumlah_kelas)]

print(f"Berdasarkan data, buah tersebut diklasifikasikan sebagai {kelas_mayoritas}.")

# Visualisasi data
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Plot penyebaran data buah
for jenis in np.unique(data[:, 2]):
    ax1.scatter(data[data[:, 2] == jenis][:, 0].astype(int),
                data[data[:, 2] == jenis][:, 1].astype(int),
                label=jenis)
ax1.set_title("Persebaran Data Buah")
ax1.set_xlabel("Berat (gram)")
ax1.set_ylabel("Diameter (cm)")
ax1.legend()

# Plot klasifikasi buah dengan K-NN
for jenis in np.unique(data[:, 2]):
    ax2.scatter(data[data[:, 2] == jenis][:, 0].astype(int),
                data[data[:, 2] == jenis][:, 1].astype(int),
                label=jenis)

# Tambahkan buah yang diprediksi dan k titik data terdekat
ax2.scatter(buah[0], buah[1], color='blue', label="Buah yang Diprediksi (Apel)", marker='x')
ax2.scatter(k_titik_data[:, 0].astype(int), k_titik_data[:, 1].astype(int), color='green', label="Nilai Titik K Terdekat")

ax2.set_title("Klasifikasi Buah dengan K-NN")
ax2.set_xlabel("Berat (gram)")
ax2.set_ylabel("Diameter (cm)")
ax2.legend()

plt.tight_layout()
plt.show()

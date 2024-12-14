import numpy as np

#contoh data buah
data = np.array([
    [150,6,'Apel'],
    [175,7,'Apel'],
    [200,8,'Apel'],
    [225,9,'Apel'],
    [250,10,'Apel'],
    [275,11,'Apel'],
    [140,5,'Jeruk'],
    [165,6,'Jeruk'],
    [190,7,'Jeruk'],
    [215,8,'Jeruk'],
    [240,9,'Jeruk'],
    [265,10,'Jeruk']
])

#buah yang akan diklafisikasikan
buah = np.array([220, 8])

#hitung jarak antara  buah yang akan diklasifikasikan dengan setiap data
jarak = np.sqrt(np.sum((data[:, :2].astype(int) - buah)**2, axis=1))
print(jarak)

#tentukan nilai k dan pilih k titik data terdekat
k = 3
idx = np.argsort(jarak)[:k]
k_titik_data = data[idx]
print(k_titik_data)

#tentukan kelas mayoritas dari k titik data terdekat 
kelas = np.unique(k_titik_data[:, 2])
jumlah_kelas = np.array([len(k_titik_data[k_titik_data[:, 2]== k])for k in kelas])
kelas_mayoritas = kelas[np.argmax(jumlah_kelas)]

print(f"berdasarkan data, buah tersbeut diklasifikasiakn sebagai {kelas_mayoritas}.")
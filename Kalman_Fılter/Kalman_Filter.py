import cv2
import numpy as np 
import matplotlib.pyplot as plt
import time

dt = 1.0 #kaç saniyede bir topun konumuna bakacağımız

total_time = 50.0 #simülasyonun toplam kaç sn süreceği
#t de zaman vektörümüz zaman dizisini oluşturuyoruz
t = np.arange(0,total_time,dt) 

"""Durum (State): Bir topu herhangi bir anda tanımlamak için ne bilmemiz gerekir? 2D düzlemde, konumunu (x ve y) ve hızını (x ve y yönlerindeki hızları) bilmemiz yeterlidir. Buna "durum vektörü" denir. Bizim durum vektörümüz şöyle olacak: [pozisyon_x, pozisyon_y, hız_x, hız_y].

Hareket Kuralı: Fizik derslerinden bildiğimiz basit bir kural var: Yeni Konum = Eski Konum + Hız * Zaman. Bu kuralı kullanarak topun her dt saniyede bir nereye gideceğini hesaplayacağız."""

#top 0,0 noktasından başlasın x yönünde 2m/s hızla gitsin

initial_state_true = np.array([0,0,2,5])

#her bir zaman adımındaki gerçek durumu tutması için bir dizi oluşturalım başlangıçta hepsi 0 olacak
#len t satır ve 4 sütun u var px py vx vy
states_true = np.zeros((len(t) ,4))

#ilk durumuzu yazıyoruz
states_true[0,:] = initial_state_true

#A durum geçiş matrisidir matrisi dum vektörü ile çarpınca 
# px_yeni = 1*px + 0*py + dt*vx + 0*vy  (px_yeni = px + vx*dt)
# py_yeni = 0*px + 1*py + 0*vx + dt*vy  (py_yeni = py + vy*dt)
# vx_yeni = 0*px + 0*py + 1*vx + 0*vy  (vx_yeni = vx) - Hız sabit
# vy_yeni = 0*px + 0*py + 0*vx + 1*vy  (vy_yeni = vy) - Hız sabit

A = np.array([[1, 0, dt, 0],
              [0, 1, 0, dt],
              [0, 0, 1, 0],
              [0, 0, 0, 1]])


#döngü ile yörüngeyi hesaplayalım

for k in range(1,len(t)):
    states_true[k,:] = A @ states_true[k-1, :]

# Ölçüm Matrisi (H): Bu matris, 4 elemanlı durum vektörümüzden (px, py, vx, vy)
# sensörün ölçebildiği 2 elemanlı ölçüm vektörünü (px, py) nasıl çıkaracağımızı söyler.
# Bu matrisi durum vektörüyle çarptığımızda:
# olcum_x = 1*px + 0*py + 0*vx + 0*vy -> sadece px'i alır
# olcum_y = 0*px + 1*py + 0*vx + 0*vy -> sadece py'yi alır

H = np.array([[1, 0, 0, 0],
              [0, 1, 0, 0]])

# Ölçüm Gürültüsü Kovaryansı (R): Bu, sensörümüzün ne kadar "gürültülü" veya
# "hatalı" olduğunu temsil eden bir matristir. Değer ne kadar büyükse, sensör
# o kadar çok hata yapar. Bizim sensörümüz x ve y yönlerinde eşit hata yapsın.

r_val = 10.0 #gürültü miktarı

R = r_val * np.eye(2) #2x2 lik matris oluşturur

# measurements: Her bir zaman adımındaki gürültülü ölçümleri saklamak için
# bir dizi oluşturalım. len(t) satır ve 2 sütundan (px, py) oluşur.
measurements = np.zeros((len(t), 2))


# Şimdi bir döngü ile her gerçek konuma gürültü ekleyerek ölçüm oluşturalım. 

for k in range(len(t)):
    #states true dan pozisyonu alıyoruz
    true_pos = H @ states_true[k,:]

    # R matrisine uygun bir gürültü üretiyoruz 
    #ortalaması 0 yani gürültü negatif ya da pozitif olabilir 

    measurement_noise = np.random.multivariate_normal(mean=np.array([0,0]) , cov = R)

    #ölçüm = gerçek pozisyon + measurment_noise

    measurements[k , :] = true_pos + measurement_noise

#çizdirme kısmı 


# plt.figure(figsize=(10, 7))
# plt.plot(states_true[:, 0], states_true[:, 1], 'g-', label='Gerçek Yörünge')
# plt.scatter(measurements[:, 0], measurements[:, 1], c='b', marker='.', label='Gürültülü Ölçümler')
# plt.title("Gerçek Yörünge ve Sensör Ölçümleri")
# plt.xlabel("X Pozisyonu")
# plt.ylabel("Y Pozisyonu")
# plt.legend()
# plt.grid(True)
# plt.axis('equal')
# plt.show()


#kalman filtresi uygulama kısmı

# Proses Gürültüsü Kovaryansı (Q): Fizik modelimizin (A matrisi) ne kadar
# güvenilir olduğunu söyler. Küçük değerler modele çok güvendiğimizi, büyük
# değerler ise modelin hatalı olabileceğini (örn. ani bir rüzgar) belirtir.

q_val = 0.05

Q = q_val * np.array([[dt**4/4, 0, dt**3/2, 0],
                     [0, dt**4/4, 0, dt**3/2],
                     [dt**3/2, 0, dt**2, 0],
                     [0, dt**3/2, 0, dt**2]])


# tahmin edilen durumları saklamak için dizin

states_estimated = np.zeros((len(t) , 4))

#filtre başlangıç değerleri
#x_hat filtrenin ilk tahmini olacak hiç bi bilgi yok o yüzden 0 veriyoruz

x_hat = np.array([0,0,0,0])


# P ise filtrenin başlangıçtaki belirsizliği (kovaryansı)

P = 100 * np.eye(4)

states_estimated[0,:] = x_hat

for k in range(1 , len(t)):
    # --- 1. TAHMİN ADIMI ---
    #Fizik modelimizi (A) kullanarak bir önceki durumdan şimdiki durumu tahmin et
    x_hat_pred = A @ x_hat
    # Belirsizliğimizi de bir sonraki adıma taşı. Modeldeki gürültü (Q) nedeniyle
    # belirsizliğimiz bu adımda biraz artar.

    P_pred = A @ P @ A.T + Q

    # 2. adım güncelleme kısmı
    #bu adımdaki gürültülü ölçüm bizim yeni kanıtımız olacak

    z = measurements[k,:]

    # Kalman kazancı K en çok önemlisi
    #bu ölçüme mi güveneyim kendime mi arasında akıllı bir denge kurar

    K = P_pred @ H.T @ np.linalg.inv(H @ P_pred @ H.T +R)

    #Yeni ölçüm ve kalman kazancını kullanarak tahmini güncelliyoruz
    #yeni tahmin = önceki tahmin +düzeltme miktarı
    #düzeltme miktarı = K* (gerçek ölçüm - beklenen ölçüm)

    x_hat = x_hat_pred + K @ ((z - H @ x_hat_pred))

    #yeni ve daha iyi tahminle belirsizliği güncelliyoruz (azaltıyoruz)

    P = (np.eye(4) - K @ H) @ P_pred

    #en iyi sonucumuzu dizinimize kaydediyoruz

    states_estimated[k,:] = x_hat





# son kısım sonuç görselleştirme ve hata analizi

#1. grafik yörünge karşılaştırması

print("\nSonuç grafikleri oluşturuluyor...")


plt.figure(figsize=(12, 8)) # Grafiğin boyutunu ayarla

plt.plot(states_true[:, 0], states_true[:, 1], 'g-', label='Gerçek Yörünge', linewidth=2)
plt.scatter(measurements[:, 0], measurements[:, 1], c='b', marker='.', label='Gürültülü Ölçümler', alpha=0.6)
plt.plot(states_estimated[:, 0], states_estimated[:, 1], 'r--', label='Kalman Filtresi Tahmini', linewidth=2)

plt.title('Kalman Filtresi ile Hareket Tahmini Karşılaştırması')
plt.xlabel('X Pozisyonu (m)')
plt.ylabel('Y Pozisyonu (m)')
plt.legend()  # Etiketleri göster
plt.grid(True) # Izgara ekle
plt.axis('equal') # Eksenleri eşit ölçekle, böylece yörünge bozuk görünmez
plt.savefig('trajectory_comparison.png') # Grafiği dosyaya kaydet
plt.show()


#son analiz kısmı

pos_true = states_true[: , :2] #gerçek durumun ilk iki sütunu

pos_estimated = states_estimated[: , :2]#tahmin edilen durumun ilk iki sütunu

#hata = gerçek -tahmin

errors = pos_true - pos_estimated

# RMSE = Hataların karelerinin ortalamasının karekökü

squared_errors = errors**2
mean_squared_errors = np.mean(squared_errors , axis= 0)
rmse = np.sqrt(mean_squared_errors)

print("\n--- Hata Analizi (RMSE) ---")
print(f"X pozisyonu için RMSE: {rmse[0]:.4f} metre")
print(f"Y pozisyonu için RMSE: {rmse[1]:.4f} metre")

# Genel pozisyon hatası
overall_rmse = np.sqrt(np.mean(rmse**2))
print(f"Genel Pozisyon RMSE: {overall_rmse:.4f} metre")

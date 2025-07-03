import cv2
import numpy as np
import os
import time
from mss import mss
from ultralytics import YOLO

# -----------------------------
# 📁 Klasör oluşturma
# -----------------------------
save_dir = 'DenemeCR'
os.makedirs(save_dir, exist_ok=True)

# -----------------------------
# 🧠 YOLO modeli yükleme
# -----------------------------
model = YOLO("best.pt")  # kendi model dosyan

# -----------------------------
# 🖥️ Ekran boyutunu belirle
# -----------------------------
mon = {'top': 100, 'left': 100, 'width': 1280, 'height': 720}
sct = mss()

# -----------------------------
# 📹 VideoWriter ayarları
# -----------------------------
fps = 20.0
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_path = os.path.join(save_dir, 'ekran_kaydi_sonuc.avi')
out = cv2.VideoWriter(output_path, fourcc, fps, (mon['width'], mon['height']))

# -----------------------------
# ⏱️ Süre Ayarı (örn. 30 saniye kayıt)
# -----------------------------
record_seconds = 30
start_time = time.time()

print("[INFO] Kayıt başladı...")

while True:
    img = np.array(sct.grab(mon))
    frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    # YOLO ile tespit yap
    results = model.predict(frame, verbose=False)

    # Sonuçları çiz
    annotated_frame = results[0].plot()

    # Videoya yaz
    out.write(annotated_frame)

    # Aynı zamanda ekranda göster (istersen)
    cv2.imshow('YOLO Ekran Kaydı', annotated_frame)

    # Çıkmak için q'ya bas veya süre dolduysa çık
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("[INFO] Kullanıcı tarafından durduruldu.")
        break
    if (time.time() - start_time) > record_seconds:
        print("[INFO] Süre doldu, kayıt tamamlandı.")
        break

# -----------------------------
# 🎬 Kayıt sonlandırma
# -----------------------------
out.release()
cv2.destroyAllWindows()

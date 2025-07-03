# 🏹 ClashRoyale Nesne Tespiti - YOLOv8n

Bu proje, **Clash Royale** oyununa özel olarak **YOLOv8n** mimarisi ile eğitilmiş, **19 sınıftan oluşan** bir nesne tespiti modelini içermektedir. Proje kapsamında ekran görüntüsünden canlı olarak nesneleri tespit eden bir sistem geliştirilmiş ve tüm süreç video olarak kaydedilmiştir.

## 🧠 Model Hakkında

- Kullanılan mimari: `YOLOv8n`
- Eğitim: Özel olarak Clash Royale oyununa ait görüntüler ile **fine-tuning**
- Sınıf Sayısı: **19 farklı oyun içi nesne**
- Model dosyası: `best.pt`

## 📊 Performans Metrikleri

Aşağıdaki metrik dosyaları ile modelin başarımını analiz edebilirsiniz:

- 📈 `PR_Curve.png` - Precision-Recall eğrisi
- 📉 `F1_curve.png` - F1-score değişimi
- 🧾 `results.png` - Eğitim & validasyon başarımı (mAP, Precision, Recall vs.)
- 🔀 `confusion_matrix.png` - Sınıflar arası karışıklık matrisi

Bu grafikler `Proje içinde` yer almaktadır.

## 📦 Gerekli Kütüphaneler

Projeyi çalıştırmak için aşağıdaki Python kütüphaneleri gereklidir:

```bash
pip install ultralytics opencv-python mss numpy

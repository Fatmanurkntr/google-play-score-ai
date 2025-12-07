# Google Play Store Uygulama Puanı Tahminleyicisi (App Rating Predictor)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Random%20Forest-green)

## Proje Hakkında

Bu proje, **MultiGroup Zero2End Machine Learning Bootcamp** kapsamında final bitirme projesi olarak geliştirilmiştir.

Mobil uygulama pazarındaki rekabetin artmasıyla birlikte, bir uygulamanın başarısını etkileyen faktörlerin belirlenmesi geliştiriciler için kritik hale gelmiştir. Bu proje, Google Play Store üzerindeki binlerce uygulamanın teknik ve etkileşim verilerini (yorum sayısı, boyut, güncelleme sıklığı vb.) analiz ederek, uygulamanın potansiyel **Kullanıcı Puanını (Rating)** tahmin eden uçtan uca bir makine öğrenmesi çözümüdür.

## Veri Seti

Proje, Kaggle platformunda bulunan ve Google Play Store'daki yaklaşık 10.800 uygulamanın verilerini içeren veri seti üzerinde geliştirilmiştir.

- **Veri Kaynağı:** [Google Play Store Apps - Kaggle](https://www.kaggle.com/datasets/lava18/google-play-store-apps)
- **Veri Boyutu:** ~10.841 Satır, 13 Değişken
- **Hedef Değişken:** Rating (1.0 - 5.0 arası kullanıcı puanı)
- **Veri Tipi:** Gerçek dünya verisi (Real-world data), kirli ve temizlik gerektiren yapıdadır.

> **Not:** Veri seti proje dizinindeki `data/` klasörü altında işlenmiş haliyle bulunmaktadır.

---

## Proje Metodolojisi ve Teknik Kararlar

Proje, ham veriden canlı ürüne giden yolda aşağıdaki teknik aşamalardan geçmiştir:

### 1. Keşifsel Veri Analizi (EDA) ve Veri Temizliği

Veri seti ham haldeyken makine öğrenmesi modelleri için uygun olmayan yapılar barındırıyordu.

- **Hatalı Veri Tespiti:** Kategori sütununda "1.9" gibi hatalı bir kaydırma (data shift) içeren satır tespit edilerek temizlendi.
- **Tip Dönüşümleri:** "Size" (Boyut), "Installs" (İndirme) ve "Price" (Fiyat) sütunlarındaki özel karakterler ('M', 'k', '+', '$') temizlenerek sayısal formata dönüştürüldü.
- **Eksik Veri Analizi:** Rating sütunundaki eksik veriler, veri dağılımını bozmamak adına başlangıçta ortalama ile dolduruldu.

### 2. Baseline (Referans) Model Kurulumu

Modelin başarısını ölçebilmek için herhangi bir özellik mühendisliği (Feature Engineering) yapılmadan ham veri ile referans modeller kuruldu.

- **Modeller:** Dummy Regressor, Linear Regression, Random Forest.
- **Sonuç:** En iyi sonucu **R2 Skoru: 0.1141** ile Random Forest verdi. Bu skor, ham verinin yetersiz olduğunu ve ciddi bir özellik mühendisliğine ihtiyaç duyulduğunu kanıtladı.

### 3. Özellik Mühendisliği (Feature Engineering)

Modelin tahmin gücünü artırmak için veriden yeni ve anlamlı öznitelikler türetildi.

- **Interaction Rate (Etkileşim Oranı):** `Reviews / Installs` formülü ile hesaplandı. İndiren kullanıcıların ne kadarının geri bildirim verdiğini ölçer.
- **Days Since Update (Güncellik):** Uygulamanın son güncellenme tarihinden bugüne geçen gün sayısı hesaplandı.
- **Smart Imputation (Akıllı Doldurma):** Uygulama boyutu (Size) eksik olan veriler, genel ortalama yerine ait oldukları **Kategorinin Medyanı** ile dolduruldu.
- **Metin ve İsim Analizi:** Uygulama başlığının uzunluğu (`Title_Length`) ve "Pro/Premium" gibi anahtar kelimelerin varlığı (`Is_Pro_App`) değişkene dönüştürüldü.
- **Clustering (Kümeleme):** K-Means algoritması ile uygulamalar 5 farklı profile ayrıldı ve bu bilgi modele `App_Cluster` özelliği olarak verildi.
- **Gürültü Temizliği:** 10'dan az yorumu olan uygulamalar "güvenilmez veri" kabul edilerek veri setinden çıkarıldı.

### 4. Model Optimizasyonu

- **Algoritma Seçimi:** Random Forest ve XGBoost algoritmaları, `RandomizedSearchCV` kullanılarak en iyi hiperparametreler ile yarıştı.
- **Sonuç:** Random Forest, **R2 Skoru: 0.2247** ile XGBoost'u geride bırakarak final model olarak seçildi.
- **Başarı Artışı:** Yapılan işlemler sonucunda model başarısı, Baseline skora (0.1141) kıyasla **%97 oranında artış** gösterdi.

### 5. Değerlendirme (Evaluation)

Modelin karar mekanizması "Feature Importance" analizi ile incelendi.

- **En Kritik Faktör:** **Interaction Rate** (Etkileşim Oranı). Model, yorum yapma oranı yüksek olan uygulamaların daha yüksek puan aldığını tespit etti.
- **İkinci Faktör:** **Days Since Update**. Uygulamanın güncel tutulması, puanı doğrudan etkileyen en önemli ikinci faktör oldu.

---

## Model Performansı

| Model                   | R2 Skoru   | MAE (Ortalama Hata) | Açıklama                      |
| :---------------------- | :--------- | :------------------ | :---------------------------- |
| **Baseline (Ham Veri)** | 0.1141     | 0.2837              | Başlangıç noktası.            |
| **Final (Tuned RF)**    | **0.2247** | **0.2630**          | Optimize edilmiş final sonuç. |

---

## İş İçgörüleri (Business Insights)

Veri analizi ve model sonuçlarına dayanarak geliştiriciler için stratejik öneriler:

1.  **Etkileşimi Artırın:** Sadece indirme sayısına odaklanmayın. Kullanıcıları yorum yapmaya teşvik eden "Bizi Puanlayın" kurguları, puanı artırmanın en etkili yoludur.
2.  **Uygulamayı Güncel Tutun:** "Days Since Update" özelliği model için çok kritiktir. 3 aydan uzun süre güncellenmeyen uygulamaların puanı düşme eğilimindedir.
3.  **Optimum İsimlendirme:** Başlık uzunluğu puanı etkilemektedir. ASO (App Store Optimization) kurallarına uygun, ne çok kısa ne çok uzun isimler tercih edilmelidir.

---

## Kurulum ve Çalıştırma

Projeyi yerel bilgisayarınızda çalıştırmak için aşağıdaki adımları izleyebilirsiniz.

**1. Repoyu Klonlayın**

````bash
git clone [https://github.com/Fatmanurkntr/google-play-score-ai.git](https://github.com/Fatmanurkntr/google-play-score-ai.git)
cd google-play-score-ai
````

**2. Gereksinimleri Yükleyin**

```bash
pip install -r requirements.txt
````

**3. Uygulamayı Başlatın**

```bash
streamlit run src/app.py
```

**Dosya Yapısı**

```bash
google-play-score-ai/
│
├── data/
│   ├── google_play_data.csv           # Temizlenmiş ham veri
│   └── processed_google_play_data.csv # Feature Engineering sonrası işlenmiş veri
├── models/
│   └── best_model.pkl                 # Eğitilmiş final model (Random Forest)
├── notebooks/
│   ├── 1_eda.ipynb                    # Keşifsel Veri Analizi
│   ├── 2_baseline.ipynb               # Temel model kurulumu ve kıyaslama
│   ├── 3_feature_engineering.ipynb    # Özellik türetimi ve dönüşümler
│   ├── 4_model_optimization.ipynb     # Hiperparametre optimizasyonu
│   └── 5_model_evaluation.ipynb       # Model değerlendirmesi ve içgörüler
├── src/
│   ├── app.py                         # Streamlit arayüz kodu (Frontend)
│   └── inference.py                   # Tahminleme ve ön işleme motoru (Backend)
├── images/                            # Proje grafikleri
├── requirements.txt                   # Gerekli kütüphaneler
└── README.md                          # Proje dokümantasyonu
```

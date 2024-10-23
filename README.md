# SAYZEK2024 - AI-SEC Heron Takımı

Bu proje, SAYZEK 2024 yarışması için AI-SEC Heron Takımı tarafından geliştirilmiş bir nesne tespit modelini içermektedir. Model, YOLOv5 kullanılarak dört farklı nesne kategorisi için özelleştirilmiştir: Bina, Yol Kesişimi, Halısaha ve Silo.

## 1. Kurulum Adımları

Proje ortamının hazırlanması ve modelin çalıştırılması için aşağıdaki adımları takip edebilirsiniz:

### 1.1 Gerekli Yazılımlar

- Python 3.8 veya üzeri
- Git
- CUDA (GPU ile çalıştırma için)

### 1.2 Repository'yi Klonlama

```bash
git clone https://github.com/BesirVelioglu/SAYZEK2024_AI-SEC-Heron.git
cd SAYZEK2024_AI-SEC-Heron
```

### 1.3 Sanal Ortam Oluşturma ve Paketlerin Kurulumu

Python paketlerini bağımsız bir ortamda kullanabilmek için sanal bir ortam oluşturun:

```bash
python -m venv venv
source venv/bin/activate  # Windows için: venv\Scriptsctivate
```

Gerekli tüm Python paketlerini `requirements.txt` dosyasından kurabilirsiniz:

```bash
pip install -r requirements.txt
```

## 2. Ortam Hazırlığı ve Gerekli Paketlerin Kurulumu

### 2.1 Paketler

Proje, aşağıdaki temel paketlere ihtiyaç duyar:

- `torch`: PyTorch ile derin öğrenme modellerini çalıştırmak için
- `opencv-python`: Görüntü işleme ve bounding box'ların çizilmesi için
- `PyYAML`: Yapılandırma dosyalarını işlemek için
- `Pillow`: Görüntülerin yüklenmesi ve işlenmesi için
- `yolov5`: YOLOv5 modellerinin yüklenmesi ve çalıştırılması için

Paketlerin kurulumu yukarıdaki komut ile `requirements.txt` dosyasından yapılabilir.

## 3. Eğitim ve Çıkarım Komutları

### 3.1 Model Eğitimi

Modeli eğitmek için `train.py` dosyasını kullanabilirsiniz. Eğitim esnasında veri seti yolları, sınıf sayısı gibi yapılandırma bilgilerini içeren bir YAML dosyası belirtmeniz gerekir.

Aşağıdaki komut ile eğitim başlatılabilir:

```bash
python src/train.py --config src/config/train_config.yaml
```

Bu komut, belirttiğiniz yapılandırma dosyasına göre eğitim sürecini başlatacaktır.

### 3.2 Modelle Çıkarım (Inference) Yapma

Eğitilen modelleri kullanarak test görüntüleri üzerinde çıkarım yapmak için aşağıdaki komutu çalıştırabilirsiniz:

```bash
python src/inference.py
```

Yukarıdaki komut, `infer_config.yaml` dosyasındaki test görüntüleri yolunu ve model dosyalarını kullanarak çıkarım işlemini başlatır. Sonuçlar, `output_images_path` dizininde görselleştirilmiş olarak kaydedilecektir.

## 4. Modelin Çalıştırılması İçin Gerekli Adımlar

1. **Model Eğitim Yapılandırması**: `train_config.yaml` dosyasını düzenleyerek veri yollarını, eğitim parametrelerini (batch size, epoch sayısı vb.) belirleyin.
2. **Eğitimi Başlatın**: Yukarıdaki eğitim komutunu çalıştırın.
3. **Modeli Kaydetme**: Eğitim sürecinde model, belirtilen dizine kaydedilecektir.
4. **Çıkarım Yapın**: `inference.py` dosyasını çalıştırarak test görüntüleri üzerinde çıkarım yapın.

## 5. Ek Notlar ve Öneriler

- **GPU Desteği**: Proje GPU ile optimize edilmiştir. Eğer GPU kullanıyorsanız, CUDA desteğinin etkin olduğundan emin olun.
- **Veri Yolları**: Eğitim ve çıkarım için kullanılan veri yollarının yapılandırma dosyalarında doğru şekilde belirtildiğinden emin olun.
- **Hyperparametre Ayarları**: Model performansını artırmak için learning rate, batch size gibi hiperparametrelerde ince ayar yapabilirsiniz.
- **Sonuçların İncelenmesi**: Çıktı görüntüleri, bounding box'lar ile birlikte `output_images_path` altında kaydedilecektir.

---

Proje ile ilgili sorularınız veya geri bildirimleriniz için lütfen bizimle iletişime geçin: mbesirvelioglu@gmail.com

sametkaras.tr@gmail.com

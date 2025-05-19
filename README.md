# Laporan Proyek Machine Learning - Laila Wulandari

## Domain Proyek

Pesatnya perkembangan teknologi digital memicu tantangan baru, yakni overstimulasi. Overstimulasi digital disebabkan oleh konsumsi media sosial, gim, dan streaming yang kini menjadi kebiasaan umum, terutama di kalangan Gen Z. Penelitian oleh Wibowo & Yulianto (2023) menyoroti dampak neuropsikologis dari overstimulasi, khususnya gangguan pada konektivitas saraf antara korteks prefrontal dan striatum yang menyebabkan penurunan kemampuan kontrol perhatian. Kondisi ini diperburuk oleh kebiasaan multitasking digital dan screen time yang tinggi.

Data dari BePresent Digital Wellness Report (2024) menunjukkan bahwa 83% Gen Z mengaku memiliki hubungan tidak sehat dengan ponsel mereka, dan 66% mengalami gangguan tidur akibat penggunaannya. Selain itu, 73% dari keseluruhan responden merasa ponsel berdampak negatif terhadap kesehatan mental mereka. Masalah ini perlu segera diatasi karena berpotensi memicu gangguan mental jangka panjang seperti kecemasan, kelelahan kognitif, hingga burnout.

Proyek ini bertujuan membangun model machine learning prediktif berbasis dataset yang mencakup aspek gaya hidup digital, seperti durasi screen time, jam tidur, tingkat stres, dan frekuensi interaksi sosial. Tujuan utamanya adalah mendeteksi risiko overstimulasi secara dini agar dapat dilakukan intervensi preventif berbasis data.

Lakilaki, E., Puri, R. M., Saputra, A. N. Z., Shawmi, A. N., Asiah, N., & Rizky, M. (2025). The phenomenological analysis of the impact of digital overstimulation on attention control in elementary school students: A study on the 'brain rot' phenomenon in the learning process. TOFEDU: The Future of Education Journal, 4(1), 265–274.

BePresent. (2024, June). BePresent 2024 digital wellness report. Retrieved from https://www.bepresentapp.com/post/bepresent-2024-digital-wellness-report

## Business Understanding

### Problem Statements

- Pesatnya konsumsi teknologi digital telah memicu fenomena overstimulasi, khususnya pada Gen Z yang memiliki screen time tinggi dan kebiasaan multitasking. Namun, belum tersedia pendekatan berbasis data untuk mendeteksi secara dini individu yang berisiko mengalami overstimulasi digital.
- Belum diketahui fitur gaya hidup digital apa saja yang paling berpengaruh dalam memicu risiko overstimulasi digital—misalnya antara screen time, kualitas tidur, tingkat stres, dan interaksi sosial.
- Diperlukan model prediktif yang mampu mengolah data gaya hidup dan memberikan hasil yang akurat serta dapat diinterpretasikan sebagai dasar intervensi preventif.

### Goals

- Mengembangkan model machine learning yang mampu memprediksi risiko overstimulasi digital berdasarkan data gaya hidup, seperti durasi screen time, kualitas tidur, dan tingkat stres.
- Mengidentifikasi faktor gaya hidup paling signifikan yang memengaruhi risiko overstimulasi digital, sehingga dapat menjadi dasar intervensi preventif yang lebih tepat sasaran.
- Membangun sistem prediktif yang tidak hanya akurat, tetapi juga mudah diinterpretasikan dan berpotensi diimplementasikan dalam bentuk aplikasi atau sistem rekomendasi.

### Solution Statement
Solusi 1 – Baseline Model Multialgoritma
Membangun model klasifikasi awal menggunakan Logistic Regression, SVM, dan KNN untuk memprediksi risiko overstimulasi digital berdasarkan variabel gaya hidup. Evaluasi dilakukan menggunakan akurasi, precision, recall, F1-score, serta confusion matrix.

Solusi 2 – Analisis Feature Importance
Mengidentifikasi fitur paling signifikan yang berkontribusi terhadap risiko overstimulasi dengan mengecek korelasi fitur dengan terget, serta menggunakan coefficients pada model Logistic Regression, untuk membantu pemahaman yang lebih dalam dan actionable insight.

Solusi 3 – Peningkatan Model melalui Hyperparameter Tuning
Melakukan tuning pada model terbaik menggunakan GridSearchCV guna mengoptimalkan kinerja model, serta menyempurnakan klasifikasi berdasarkan threshold dari ROC Curve.


## Data Understanding

Dataset yang digunakan dalam proyek ini adalah Overstimulation Behavior and Lifestyle Dataset, yang tersedia secara publik di Kaggle melalui tautan berikut: https://www.kaggle.com/datasets/miadul/overstimulation-behavior-and-lifestyle-dataset. Dataset ini bertujuan untuk menangkap hubungan antara gaya hidup digital, kesehatan mental, dan kemungkinan overstimulasi. Dataset terdiri dari 2000 baris data dan 20 kolom fitur, termasuk fitur target Overstimulated. Seluruh data bersifat numerik dan telah melalui proses pembersihan dasar.

### Variabel-variabel pada Overstimulation Behavior and Lifestyle Dataset adalah sebagai berikut:

- Age: Usia individu (bilangan bulat antara 18 dan 60)
- Sleep_Hours: Jumlah jam tidur yang didapatkan individu per hari (mengambang antara 3 dan 10)
- Screen_Time: Total waktu layar (dalam jam) yang dihabiskan individu pada perangkat per hari (mengambang antara 1 dan 12)
- Stress_Level: Tingkat stres yang dilaporkan sendiri dalam skala 1 hingga 10 (1 = stres rendah, 10 = stres tinggi)
- Noise_Exposure: Frekuensi paparan tingkat kebisingan tinggi (bilangan bulat antara 0 dan 5)
- Social_Interaction: Jumlah interaksi sosial per hari (bilangan bulat antara 0 dan 10)
- Work_Hours: Jumlah jam kerja per hari (bilangan bulat antara 4 dan 15)
- Exercise_Hours: Jumlah jam yang dihabiskan untuk berolahraga per hari (mengambang antara 0 dan 3)
- Caffeine_Intake: Jumlah cangkir minuman berkafein yang dikonsumsi per hari (bilangan bulat antara 0 dan 5)
- Multitasking_Habit: Apakah individu cenderung melakukan banyak tugas (biner: 0 = Tidak, 1 = Ya)
- Anxiety_Score: Skor kecemasan yang dilaporkan sendiri pada skala 1 hingga 10 (1 = kecemasan rendah, 10 = kecemasan tinggi)
- Depression_Score: Skor depresi yang dilaporkan sendiri pada skala 1 hingga 10 (1 = depresi rendah, 10 = depresi tinggi)
- Sensory_Sensitivity: Sensitivitas terhadap input sensorik (0 = sensitivitas rendah, 4 = sensitivitas tinggi)
- Meditation_Habit: Apakah individu mempraktikkan meditasi atau perhatian penuh (biner: 0 = Tidak, 1 = Ya)
- Overthinking_Score: Skor yang dilaporkan sendiri mengenai seberapa banyak seseorang berpikir berlebihan (1 = rendah, 10 = tinggi)
- Irritability_Score: Skor iritabilitas yang dilaporkan sendiri pada skala 1 hingga 10 (1 = iritabilitas rendah, 10 = iritabilitas tinggi)
Sakit kepala
- Overstimulated: Kolom target, di mana 1 menunjukkan bahwa individu tersebut mengalami overstimulasi, dan 0 menunjukkan bahwa mereka tidak mengalami overstimulasi.

### Exploratory Data Analysis yang dilakukan:
- Melihat 5 Data Pertama untuk memeriksa contoh data untuk memahami struktur kolom.
- Mengecek tipe data dan keberadaan nilai null.
- Melakukan statistik deskriptif dengan menganalisis mean, std, min, max, dan quartile untuk memahami distribusi data.
- Melakukan visualisasi distribusi target untuk menampilkan sebaran kelas target.
- Melakukan visualisasi korelasi semua fitur terhadap fitur target untuk menentukan fitur yang layak dijadikan prioritas.

## Data Preparation
Dilakukan beberapa teknik data preparation untuk memastikan data siap digunakan dalam pemodelan. Tahap ini bertujuan untuk meningkatkan kualitas data, menghindari bias, dan memastikan model yang dibangun memiliki performa yang baik. Berikut tahapannya:

- Melakukan pemilihan fitur yang berkorelasi dengan fitur target berdasarkan hasil heatmap korelasi fitur. Dilakukan dengan menginisialisasi variable features dengan fitur terpilih (Screen_Time, Sleep_Hours, dan Stress_Level) dan menetapkan target yaitu Overstimulated. Hal ini bertujuan untuk memastikan... 
- Membagi data menjadi training set dan testing set. Dilakukan menggunakan train_test_split dari sklearn.model_selection. Bertujuan untuk mengevaluasi performa model secara adil pada data yang belum pernah dilihat sebelumnya.
- Standarisasi data menggunakan StandardScaler dari sklearn.preprocessing untuk menormalkan fitur numerik. Bertujuan untuk mempercepat konvergensi model dan meningkatkan akurasi.

## Modeling
Pada tahap ini menggunakan 3 algoritma klasifikasi:  Logistic Regression, Support Vector Machine (SVM), dan K-Nearest Neighbors (KNN) dengan hyperparameter tuning 
Tahapan ini membahas mengenai model machine learning yang digunakan untuk menyelesaikan permasalahan. Anda perlu menjelaskan tahapan dan parameter yang digunakan pada proses pemodelan.

1. Logistic Regression: 
- Digunakan karena cocok untuk masalah klasifikasi biner seperti prediksi overstimulasi digital, di mana kita ingin memahami pengaruh masing-masing fitur terhadap hasil prediksi. 
- Parameter yang Digunakan:
  - random_state=42: Untuk memastikan hasil yang konsisten/reproducible.
  - penalty='l2' (regularisasi untuk menghindari overfitting).
  - solver='lbfgs' (algoritma optimasi default untuk dataset kecil/medium).
- Kelebihan:
  - Cepat dalam pelatihan dan prediksi.
  - Output berupa probabilitas, memudahkan interpretasi.
  - Tidak memerlukan tuning parameter yang kompleks.
- Kekurangan:
  - Mengasumsikan hubungan linear antara fitur dan log-odds.
  - Sensitif terhadap outlier dan multikolinearitas.
  - Kurang efektif untuk hubungan non-linear dalam data.
2. Support Vector Machine:
- Dipilih karena kemampuannya menangani data high-dimensional dan efektif untuk kasus di mana margin pemisah antar kelas jelas. Cocok untuk data yang mungkin tidak terpisah secara linear.
- Parameter yang Digunakan:
  - probability=True: Untuk mengaktifkan prediksi probabilitas.
  - random_state=42: Reproducibility.
  - C=1.0 (parameter regularisasi).
  - kernel='rbf' (default untuk non-linear separation).
- Kelebihan:
  - Efektif pada data high-dimensional.
  - Tahan terhadap overfitting dengan parameter C yang tepat.
  - Fleksibel dengan berbagai kernel (linear, RBF, dll.).
- Kekurangan:
  - Komputasi lebih lambat untuk dataset besar.
  - Membutuhkan tuning parameter (C, kernel) untuk hasil optimal.
  - Kurang interpretatif dibanding Logistic Regression.
K-Nearest Neighbor dengan hyperparameter tuning:
- Dipilih karena kemampuannya menangani pola non-linear tanpa asumsi distribusi data. Hyperparameter tuning (n_neighbors) dilakukan untuk mengoptimalkan performa.
- Parameter yang Digunakan:
  - GridSearchCV untuk mencari n_neighbors terbaik (range 3–20).
  - n_neighbors=5 (nilai optimal dari pencarian).
  - cv=5: 5-fold cross-validation untuk evaluasi robust.
- Kelebihan:
  - Tidak memerlukan pelatihan eksplisit (lazy learning).
  - Fleksibel untuk hubungan non-linear dalam data.
  - Mudah diimplementasikan.
- Kekurangan:
  - Komputasi mahal untuk data besar (karena menyimpan semua data).
  - Sensitif terhadap skala data (harus dinormalisasi).
  - Performa buruk jika banyak fitur tidak relevan.
## Evaluation
Untuk mengukur kinerja model klasifikasi risiko overstimulasi digital, digunakan beberapa metrik evaluasi utama yang relevan dengan konteks masalah:
1. Classification Report (meliputi):
- Precision: Kemampuan model tidak memprediksi false positive
- Recall: Kemampuan model menemukan semua positive cases
- F1-score: Rata-rata harmonik precision dan recall
- Accuracy: Proporsi prediksi benar secara keseluruhan
2. Confusion Matrix:
- Menampilkan TP, FP, TN, FN secara visual
3. ROC AUC Score:
- Mengukur kemampuan model membedakan antara kelas
- Nilai 1 = sempurna, 0.5 = random guessing


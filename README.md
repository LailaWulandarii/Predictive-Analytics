# Laporan Proyek Machine Learning - Laila Wulandari

## Domain Proyek

Pesatnya perkembangan teknologi digital memicu tantangan baru, yakni overstimulasi. Overstimulasi digital disebabkan oleh konsumsi media sosial, gim, dan streaming yang kini menjadi kebiasaan umum, terutama di kalangan Gen Z. Penelitian oleh Wibowo & Yulianto (2023) menyoroti dampak neuropsikologis dari overstimulasi, khususnya gangguan pada konektivitas saraf antara korteks prefrontal dan striatum yang menyebabkan penurunan kemampuan kontrol perhatian. Kondisi ini diperburuk oleh kebiasaan multitasking digital dan screen time yang tinggi.

Data dari BePresent Digital Wellness Report (2024) menunjukkan bahwa 83% Gen Z mengaku memiliki hubungan tidak sehat dengan ponsel mereka, dan 66% mengalami gangguan tidur akibat penggunaannya. Selain itu, 73% dari keseluruhan responden merasa ponsel berdampak negatif terhadap kesehatan mental mereka. Masalah ini perlu segera diatasi karena berpotensi memicu gangguan mental jangka panjang seperti kecemasan, kelelahan kognitif, hingga burnout.

Proyek ini bertujuan membangun model machine learning prediktif berbasis dataset yang mencakup aspek gaya hidup digital, seperti durasi screen time, jam tidur, tingkat stres, dan frekuensi interaksi sosial. Tujuan utamanya adalah mendeteksi risiko overstimulasi secara dini agar dapat dilakukan intervensi preventif berbasis data.

Lakilaki, E., Puri, R. M., Saputra, A. N. Z., Shawmi, A. N., Asiah, N., & Rizky, M. (2025). The phenomenological analysis of the impact of digital overstimulation on attention control in elementary school students: A study on the 'brain rot' phenomenon in the learning process. TOFEDU: The Future of Education Journal, 4(1), 265–274.

BePresent. (2024, June). BePresent 2024 digital wellness report. Retrieved from https://www.bepresentapp.com/post/bepresent-2024-digital-wellness-report

## Business Understanding

### Problem Statements

- Pesatnya konsumsi media digital (sosial media, gim, streaming) dan kebiasaan multitasking telah mengganggu konektivitas saraf prefrontal-striatum, menyebabkan penurunan kontrol perhatian. Namun, belum ada alat prediktif berbasis data untuk mendeteksi risiko ini secara dini.
- Data BePresent (2024) menunjukkan 83% Gen Z memiliki ketergantungan tidak sehat pada ponsel, dengan 66% mengalami gangguan tidur dan 73% melaporkan dampak negatif pada kesehatan mental. Belum diketahui faktor gaya hidup digital mana (screen time, tidur, stres) yang paling kritis memicu overstimulasi.
- Overstimulasi berpotensi memicu kecemasan, kelelahan kognitif, dan burnout. Diperlukan model prediktif yang akurat dan terinterpretasi untuk merancang intervensi preventif.

### Goals

- Mengembangkan model klasifikasi untuk memprediksi risiko overstimulasi berdasarkan fitur seperti screen time, jam tidur, tingkat stres, dan interaksi sosial, sesuai temuan neuropsikologis dan data digital wellness.
- Menganalisis korelasi fitur untuk menentukan prioritas intervensi. 
- Menghasilkan sistem prediktif yang tidak hanya akurat.
 
### Solution Statement
Solusi 1 – Baseline Model dengan Pendekatan Multialgoritma
  - Algoritma: Logistic Regression (interpretabilitas), SVM (handling non-linearitas), dan KNN (tuning jarak fitur).
  - Evaluasi: Fokus pada precision-recall (karena data imbalance) dan ROC AUC untuk mengukur kemampuan membedakan kelas risiko.

Solusi 2 – Analisis Feature Importance
Mengidentifikasi fitur paling signifikan yang berkontribusi terhadap risiko overstimulasi dengan mengecek korelasi fitur dengan terget menggunakan heatmap korelasi dan coefficients pada model Logistic Regression, untuk membantu pemahaman yang lebih dalam dan actionable insight.

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
- Headache_Frequency: Seberapa sering individu mengalami sakit kepala (bilangan bulat antara 0 dan 7, mewakili frekuensi dalam seminggu)
- Sleep_Quality: Kualitas tidur individu dalam skala 1 hingga 4 (1 = buruk, 4 = sangat baik)
- Tech_Usage_Hours:Jumlah jam yang dihabiskan untuk menggunakan teknologi (perangkat, komputer, dll.) per hari (mengambang antara 1 dan 10)
- Overstimulated: Kolom target, di mana 1 menunjukkan bahwa individu tersebut mengalami overstimulasi, dan 0 menunjukkan bahwa mereka tidak mengalami overstimulasi.

### Exploratory Data Analysis yang dilakukan:
- Melihat 5 Data Pertama untuk memeriksa contoh data untuk memahami struktur kolom.
- Mengecek tipe data dan keberadaan nilai null. Hasilnya tidak ada data kosong.
- Melakukan statistik deskriptif dengan menganalisis mean, std, min, max, dan quartile untuk memahami distribusi data. Berikut merupakan detail hasilnya.
  
    - Usia responden 18–59 tahun, dengan rata-rata 38,7.
    - Durasi tidur 3–10 jam/hari, dengan rata-rata 6,5, menunjukkan potensi kurang tidur.
    - Screen time 1–12 jam/hari, dengan rata-rata 6,4 dan tech usage 1–10 jam/hari dengan rata-rata 5,5 mengindikasikan paparan teknologi tinggi.
    - Stres level 1–9 dengan rata-rata 5 dan skor mental health 1–9 dengan rata-rata 5 menunjukkan kondisi moderat.
    - Jam kerja 9 jam/hari dengan olahraga hanya 1,5 jam/hari mencerminkan gaya hidup tidak seimbang.
    - Kualitas tidur mayoritas level 2–3/5 dengan rata-rata 2,3 dengan beberapa kasus sakit kepala intens.
- Melakukan visualisasi distribusi target untuk menampilkan sebaran kelas target. Hasilnya data memiliki persebaran seperti: 725 data yang tidak overstimulated dan 1275 data overstimulated.
- Melakukan visualisasi korelasi semua fitur terhadap fitur target untuk menentukan fitur yang layak dijadikan prioritas. Hasilnya terdapat 3 fitu (Screen_Time, Sleep_Hours, dan Stress_Level) yang berkorelasi dengan target, yang nanti akan dipilih untuk proses selanjutnya.

## Data Preparation
Dilakukan beberapa teknik data preparation untuk memastikan data siap digunakan dalam pemodelan. Tahap ini bertujuan untuk meningkatkan kualitas data, menghindari bias, dan memastikan model yang dibangun memiliki performa yang baik. Berikut tahapannya:
- Melakukan pemilihan fitur yang berkorelasi dengan fitur target berdasarkan hasil heatmap korelasi fitur. Dilakukan dengan menginisialisasi variable features dengan fitur terpilih (Screen_Time, Sleep_Hours, dan Stress_Level) dan menetapkan target yaitu Overstimulated. Hal ini bertujuan untuk memastikan semua fitur yang akan digunakan berarti untuk kolom target.
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
3. K-Nearest Neighbor dengan hyperparameter tuning:
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
1. Classification Report:
  - Precision:
    Mengukur seberapa baik model dalam memprediksi kelas positif tanpa menghasilkan terlalu banyak false positives.
     ![image](https://github.com/user-attachments/assets/ce2c2575-2213-426c-bdc2-334c1c8171e1)
  - Recall: 
    Mengukur seberapa baik model dalam menemukan semua positive cases, yaitu sejauh mana model dapat menangkap seluruh contoh positif.
    ![image](https://github.com/user-attachments/assets/76bf8fe4-98bb-4cfd-bf88-b81ec0945c01)
  - F1-score: 
    Merupakan rata-rata harmonik antara precision dan recall, memberikan keseimbangan antara keduanya.
    ![image](https://github.com/user-attachments/assets/d631b081-b8c3-4982-a9af-0d838a2f7539)
  - Accuracy:
    Mengukur proporsi keseluruhan prediksi yang benar dibandingkan dengan total data.
     ![image](https://github.com/user-attachments/assets/bad0aa15-63c5-4212-83c5-600e855c29dd)
2. Confusion Matrix, untuk menampilkan jumlah prediksi yang dilakukan oleh model dalam empat kategori utama:
  - True Positive (TP): Kasus positif yang diprediksi benar oleh model.
  - False Positive (FP): Kasus negatif yang salah diprediksi sebagai positif.
  - True Negative (TN): Kasus negatif yang diprediksi benar oleh model.
  - False Negative (FN): Kasus positif yang salah diprediksi sebagai negatif.
3. ROC AUC Score:
  - ROC (Receiver Operating Characteristic) Curve menunjukkan hubungan antara True Positive Rate (TPR) dan False Positive Rate (FPR) untuk berbagai nilai ambang batas. AUC (Area Under Curve) mengukur kemampuan model dalam membedakan antara kelas positif dan negatif
  - Nilai  berkisar antara:
    - 1.0 → Model sempurna dalam membedakan antara kelas.
    - 0.5 → Model hanya melakukan tebakan acak.
    - < 0.5 → Model lebih buruk daripada tebakan acak.
4. Interpretasi Koefisien Logistic Regression:
  - Koefisien menunjukkan seberapa besar pengaruh setiap fitur terhadap probabilitas prediksi kelas positif. Dalam Logistic Regression, koefisien menunjukkan pengaruh setiap fitur terhadap probabilitas kelas positif menggunakan log-odds.
  - Koefisien positif berarti dapat meningkatkan peluang hasil positif, sebaliknya koefisien negatif dapat mengurangi peluang hasil positif.
  - Probabilitas dihitung dengan rumus: ![image](https://github.com/user-attachments/assets/be66763b-4c08-4f5b-84f9-233e64d9ed77)


Berikut merupakan hasil evaluasi model:
1. Logistic Regression
   
   ![image](https://github.com/user-attachments/assets/afc640ff-196f-4976-ba71-8c6b7a768dc0)
2. Support Vector Machine
   
   ![image](https://github.com/user-attachments/assets/3b7cd9c0-aa78-4b39-8921-e36e05979441)
3. K-Nearest Neighbor
   
   ![image](https://github.com/user-attachments/assets/69ac8f91-04e6-47d7-a11b-5555ce2a4906)
4. Hasil interpretasi koefisien Logistic Regression:

   ![image](https://github.com/user-attachments/assets/e76d15f0-45d0-491e-a7d5-80c47a07ec27)

   
Berdasarkan hasil evaluasi tersebut dapat ditarik kesimpulan:
- Model KNN dengan k=9 merupakan model terbaik karena memiliki akurasi dan ROC AUC tertinggi, serta memilliki recall hampir sempurna.
- Model SVM menunjukkan performa yang sangat kuat, namun sedikit lebih rendah dari KNN.
- Model Logistic Regression masih memungkinkan untuk digunakan walaupun performanya lebih rendah dari kedua model lainnya, hal ini dapat disebabkan karena model kurang cocok dalam menangkap pola data non-linear.
- Berdasarkan hasil interpretasi koefisien Logistic Regression sebagai berikut:
    - Screen_Time → Semakin tinggi waktu layar, semakin besar kemungkinan hasil positif.
    - Stress_Level → Semakin tinggi tingkat stres, semakin besar kemungkinan hasil positif.
    - Sleep_Hours → Semakin lama tidur, semakin kecil kemungkinan hasil positif.
 
Berdasarkan keseluruhan tahapan proyek yang telah dilakukan, dapat disimpulkan bahwa  durasi screen time, kualitas tidur, dan tingkat stres adalah faktor gaya hidup paling signifikan yang mempengaruhi risiko overstimulasi digital. Hasil ini diharapkan dapat digunakan sebagai dasar langkah preventif, seperti membatasi screen time, meningkatkan kualitas tidur, serta menerapkan strategi manajemen stress untuk menjaga keseimbangan digital dan kesehatan mental.

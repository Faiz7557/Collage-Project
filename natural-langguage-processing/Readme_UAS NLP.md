📘 README – UAS NLP

Notes: Seluruh file kami letakkan pada gdrive pada link yang terdapat pada laporan, atau dapat diakses sebagai berikut:
https://drive.google.com/drive/folders/1XRUU8LmvtwSR6Y7MuuAs2-34vJ0AGBeR?usp=sharing



1\. Identitas Kelompok



Nomor Kelompok: Kelompok A

Nama Anggota Kelompok:

Radam Gumelar: 164231032

Raafa Agna Rasyada: 164231043

Faiz Iqbal I’tishom: 164231059

Mohammad Faizal Aprilianto: 164231095

Muhammad Rajif Al Farikhi: 162112133008



Judul Project: 



2\. Deskripsi Singkat Project

Project ini bertujuan untuk melakukan pemrosesan dan analisis teks (Natural Language Processing) menggunakan tiga pendekatan utama, yaitu:



a. Rule-Based NLP

b. Machine Learning

c. Deep Learning



Setiap pendekatan diimplementasikan dalam notebook terpisah untuk memudahkan eksperimen, evaluasi, dan perbandingan performa model.





3\. Daftar File



'rule\_based\_nlp\_uas.ipynb' → Implementasi NLP berbasis aturan (rule-based)

'Model Machine Learning\_UAS NLP\_Kelompok\_A.ipynb' → Implementasi model Machine Learning

'Model Deep Learning\_UAS NLP\_Kelompok A.ipynb' → Implementasi model Deep Learning



**4. Library yang Digunakan**



Berikut adalah library utama yang digunakan dalam project ini:

🔹 Library Umum

`numpy`

`pandas`

`re`

`os`



🔹 Natural Language Processing

`nltk`

`Sastrawi`

`scikit-learn`



🔹 Machine Learning

`scikit-learn`

`TfidfVectorizer`

`CountVectorizer`

`train\_test\_split`

&nbsp;Model klasifikasi (Naive Bayes, Logistic Regression, SVM, dll)



🔹 Deep Learning



`tensorflow` / `keras`

`torch` (jika digunakan)

`transformers` (jika menggunakan model berbasis Transformer)



🔹Visualisasi (Opsional)



matplotlib`

seaborn`



5\. Tata Cara Penggunaan Code



5.1 Persiapan Environment



A. Pastikan Python versi \*\*3.8 atau lebih baru\*\*.

B. Install seluruh library yang dibutuhkan menggunakan perintah berikut:



```bash pip install numpy pandas scikit-learn nltk sastrawi matplotlib seaborn tensorflow torch transformers```



C. Jalankan Jupyter Notebook:



``` bash jupyter notebook ```



5.2 Menjalankan Rule-Based NLP



A. Buka file `rule\_based\_nlp\_uas.ipynb`.

B. Jalankan seluruh cell secara berurutan dari atas ke bawah.

C. Pastikan dataset berada pada path yang sesuai dengan yang tertulis di dalam notebook.

D. Hasil klasifikasi akan ditampilkan pada output notebook.



5.3 Menjalankan Model Machine Learning



A. Buka file `Model Machine Learning\_UAS NLP\_Kelompok\_A.ipynb`.

B. Jalankan tahap preprocessing teks (cleaning, tokenizing, stopword removal, stemming).

C. Jalankan proses ekstraksi fitur (TF-IDF / Count Vectorizer).

D. Latih model Machine Learning yang tersedia.

E. Evaluasi model menggunakan metrik seperti \*\*accuracy, precision, recall, dan F1-score\*\*.



5.4 Menjalankan Model Deep Learning



A. Buka file `Model Deep Learning\_UAS NLP\_Kelompok A.ipynb`.

B. Pastikan library Deep Learning telah terinstall dengan benar.

C. Jalankan preprocessing data dan tokenisasi.

D. Jalankan proses training model.

E. Evaluasi performa model menggunakan data validasi atau data uji.



6\. Tata Cara Penggunaan Aplikasi



Bagian ini menjelaskan langkah-langkah menjalankan \*\*aplikasi NLP\*\* yang menggunakan model \*\*PyTorch / Transformers\*\* untuk klasifikasi berbasis Machine Learning.



6.1 Persyaratan Sistem



\*\*Python versi 3.11.9 (yang digunakan)

Sistem Operasi: Windows / Linux / macOS

Disarankan menggunakan virtual environment untuk menghindari konflik dependensi



6.2 Instalasi dan Setup Environment



A. Pastikan Python 3.11.9 telah terpasang:



```bash python --version```



B. Buat virtual environment:



bash py -3.11 -m venv venv



atau



python -m venv venv



3\. Aktifkan virtual environment:



A. Windows



```bash .\\venv\\Scripts\\activate```



B. Linux / macOS



```bash source venv/bin/activate```



C. Upgrade pip:



```bash python -m pip install --upgrade pip```



D. Install seluruh dependency:



```bash pip install -r requirements.txt```



6.3 Menjalankan Aplikasi



1\. Pastikan file utama aplikasi tersedia, yaitu `app.py`.

2\. Jalankan aplikasi dengan perintah:



```bash python app.py```



3\. Aplikasi akan berjalan dan siap menerima input teks sesuai dengan fungsionalitas yang telah diimplementasikan (misalnya klasifikasi SDGs atau topik tertentu).



6.4 Model yang Digunakan



Aplikasi \*\*tidak lagi menggunakan model legacy berbasis Joblib\*\*, yaitu:



&nbsp; `SDG\_Final\_Pipeline.joblib`

&nbsp; `ExpertRuleSDG.joblib

Pastikan path dataset sudah sesuai dengan direktori lokal masing-masing.

Disarankan menggunakan \*\*Google Colab atau Kaggle\*\* jika mengalami keterbatasan resource pada perangkat lokal.

Semua notebook dijalankan secara \*\*sequential\*\* untuk menghindari error dependensi.



7\. Penutup



README ini dibuat sebagai panduan penggunaan dan dokumentasi project UAS NLP. Diharapkan dapat membantu dalam proses evaluasi, presentasi, maupun pengembangan lanjutan.



✍️ Kelompok A – UAS Natural Language Processing




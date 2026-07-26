# Framework Transformer-LLM untuk Aspect-Level Sentiment (SemEval 2026)

## Ringkasan
Proyek ini membandingkan dua pendekatan yang secara fundamental berbeda untuk tugas Dimensional Aspect Sentiment Triplet Extraction (DimASTE), yang diperkenalkan pada SemEval 2026 Task 3 Track A Subtask 2. DimASTE mengekstraksi triplet (aspek, opini, valensi-arousal) dari teks ulasan, melampaui label positif/negatif/netral konvensional untuk menggambarkan sentimen dalam skala emosi yang kontinu. Pekerjaan ini didokumentasikan dalam paper "Perbandingan Pendekatan BERT dan LLM untuk Dimensional Aspect Sentiment Triplet Extraction."

## Tujuan
- Membandingkan pipeline berbasis RoBERTa dengan LLM yang telah di-instruction-tuning (Qwen2.5-1.5B-Instruct) pada tugas DimASTE.
- Mengevaluasi kedua pendekatan pada dua domain (ulasan restoran dan laptop) menggunakan metrik resmi Continuous F1 (cF1).
- Mengidentifikasi arsitektur mana yang lebih andal dan lebih praktis untuk ekstraksi sentimen multi-bagian yang terstruktur.

## Data
- Dataset SemEval 2026 Task 3 Track A Subtask 2 (subset bahasa Inggris), disediakan oleh penyelenggara dalam format JSON Lines.
- Domain restoran: 2.284 sampel, 3.598 triplet setelah deduplikasi.
- Domain laptop: 4.076 sampel, 5.751 triplet setelah deduplikasi.
- Setiap sampel berlabel kuadruplet (Aspect, Opinion, Category, Valence#Arousal).

## Metodologi
1. Pembersihan dan validasi data: deduplikasi pada level triplet dan level sampel, penghapusan entri yang tidak valid atau kosong, serta normalisasi yang konsisten pada kedua domain.
2. Pipeline berbasis RoBERTa: sequence labeling untuk ekstraksi span aspek/opini, tahap pairing dan klasifikasi, serta tahap regresi untuk skor valensi-arousal, dievaluasi dengan 5-fold cross-validation.
3. Pipeline berbasis LLM: instruction-based prompting dengan Qwen2.5-1.5B-Instruct, di-fine-tune menggunakan QLoRA dan framework Unsloth, dilanjutkan dengan tiga percobaan ablasi yang memvariasikan desain prompt dan hyperparameter.
4. Evaluasi menggunakan Continuous F1 (cF1), Mean Absolute Error, dan korelasi Pearson untuk prediksi valence dan arousal.

## Temuan Utama
- Qwen2.5-1.5B (Trial a) mencapai rata-rata cF1 sebesar 0,6042, jauh mengungguli RoBERTa-base yang memperoleh rata-rata 0,3540 (laptop) dan 0,4489 (restoran).
- Keunggulan LLM berasal dari kemampuannya memprediksi entitas dan skor valensi-arousal secara serentak dalam satu urutan output, sehingga menghilangkan error propagation yang rentan terjadi pada pipeline pairing diskret.
- Prompt yang lebih detail (Trial b) maupun pengaturan LoRA/training yang lebih agresif (Trial c) justru menurunkan kinerja dibandingkan prompt baseline yang lebih sederhana, menunjukkan bahwa penambahan kompleksitas tidak selalu meningkatkan kualitas ekstraksi untuk ukuran model ini.

## Relevansi Bisnis/Praktis
Pipeline DimASTE yang andal memungkinkan pelaku bisnis melampaui pelabelan positif/negatif sederhana menuju pemahaman yang lebih rinci tentang aspek spesifik mana (misalnya kecepatan pengiriman, keamanan bahan, kualitas build) yang memicu emosi pelanggan dan seberapa intens emosi tersebut, sehingga mendukung keputusan perbaikan produk dan layanan yang lebih terarah.

## Tools & Libraries
Python, PyTorch, Hugging Face Transformers, Unsloth, QLoRA, RoBERTa-base, Qwen2.5-1.5B-Instruct.

## Isi Repository
- `DimASTE_dataset.jsonl` — dataset training yang sudah dibersihkan (bahasa Inggris, domain restoran dan laptop).
- `AspectSentimentExtraction.ipynb` — notebook pipeline lengkap (pipeline RoBERTa, fine-tuning LLM, evaluasi).
- `README.md` / `README_ID.md` — dokumentasi proyek dalam bahasa Inggris dan Indonesia.

## Keterbatasan
Hasil LLM dilaporkan berdasarkan evaluasi pada set data training penuh, bukan pada test set terpisah atau cross-validation, sehingga tidak mengukur generalisasi model seketat evaluasi 5-fold yang digunakan pada RoBERTa. Inferensi LLM juga membutuhkan biaya komputasi yang relatif besar dibandingkan waktu training.

---------------------------------------------------------------------------------

# Transformer-LLM Framework for Aspect-Level Sentiment (SemEval 2026)

## Overview
This project compares two fundamentally different approaches to Dimensional Aspect Sentiment Triplet Extraction (DimASTE), a task introduced in SemEval 2026 Task 3 Track A Subtask 2. DimASTE requires extracting (aspect, opinion, valence-arousal) triplets from review text, going beyond conventional positive/negative/neutral labels to describe sentiment on a continuous emotional scale. The work is documented in the paper "Perbandingan Pendekatan BERT dan LLM untuk Dimensional Aspect Sentiment Triplet Extraction."

## Objectives
- Compare a RoBERTa-based pipeline against an instruction-tuned LLM (Qwen2.5-1.5B-Instruct) on the DimASTE task.
- Evaluate both approaches across two domains (restaurant and laptop reviews) using the official Continuous F1 (cF1) metric.
- Identify which architecture is more reliable and more practical for structured, multi-part sentiment extraction.

## Data
- SemEval 2026 Task 3 Track A Subtask 2 dataset (English subset), provided by the organizers in JSON Lines format.
- Restaurant domain: 2,284 samples, 3,598 deduplicated triplets.
- Laptop domain: 4,076 samples, 5,751 deduplicated triplets.
- Each sample is labeled with quadruplets (Aspect, Opinion, Category, Valence#Arousal).

## Methodology
1. Data cleaning and validation: triplet-level and sample-level deduplication, removal of invalid or empty entries, and consistent normalization across both domains.
2. RoBERTa-based pipeline: sequence labeling for aspect/opinion span extraction, a pairing-and-classification stage, and a regression stage for valence-arousal scores, evaluated with 5-fold cross-validation.
3. LLM-based pipeline: instruction-based prompting with Qwen2.5-1.5B-Instruct, fine-tuned via QLoRA and the Unsloth framework, followed by three ablation trials varying prompt design and hyperparameters.
4. Evaluation using Continuous F1 (cF1), Mean Absolute Error, and Pearson correlation for valence and arousal predictions.

## Key Findings
- Qwen2.5-1.5B (Trial a) achieved an average cF1 of 0.6042, clearly outperforming RoBERTa-base, which averaged 0.3540 (laptop) and 0.4489 (restaurant).
- The LLM's advantage came from predicting entities and valence-arousal scores jointly in a single output sequence, removing the error propagation that discrete pairing pipelines are prone to.
- More detailed prompts (Trial b) and more aggressive LoRA/training settings (Trial c) both reduced performance relative to the simpler baseline prompt, indicating that added complexity did not translate into better extraction quality for this model size.

## Business/Practical Relevance
A reliable DimASTE pipeline enables businesses to move beyond simple positive/negative tagging toward a nuanced view of which specific aspects (e.g., delivery speed, ingredient safety, build quality) drive customer emotion and at what intensity, supporting more targeted product and service improvement decisions.

## Tools & Libraries
Python, PyTorch, Hugging Face Transformers, Unsloth, QLoRA, RoBERTa-base, Qwen2.5-1.5B-Instruct.

## Repository Contents
- `DimASTE_dataset.jsonl` — cleaned training dataset (English, restaurant and laptop domains).
- `AspectSentimentExtraction.ipynb` — full pipeline notebook (RoBERTa pipeline, LLM fine-tuning, evaluation).
- `README.md` / `README_ID.md` — project documentation in English and Indonesian.

## Limitations
LLM results are reported on the full training set rather than a held-out test set or cross-validation, which does not measure generalization as rigorously as the 5-fold evaluation used for RoBERTa. LLM inference is also computationally expensive relative to training time.

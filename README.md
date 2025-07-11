# 🎬 Multimodal IMDB Analysis with TensorFlow  
**Evaluating CNN and LSTM Models for Film Genre Classification** 

---

## 📌 Overview

This project explores a **multimodal deep learning approach** to film genre classification using data from the Internet Movie Database (IMDB). It leverages two different models:

- A **Convolutional Neural Network (CNN)** for image-based genre classification from film posters
- A **Long Short-Term Memory (LSTM)** network for text-based classification using film overviews

Both models were built using **TensorFlow** and **Keras**, with a focus on **multi-label classification**, **GPU acceleration**, and **performance evaluation**.

---

## 🧠 Objectives

- Build and compare deep learning models on different modalities (image vs text)
- Explore model strengths in handling genre diversity and imbalance
- Evaluate performance based on precision, recall, and loss trends
- Derive insights into architecture-specific strengths and limitations

---

## 🛠️ Technologies Used

- Python
- TensorFlow & Keras
- Jupyter Notebooks
- GPU acceleration (Google Colab)
- NumPy, Matplotlib

---

## 🗃️ Data Preparation

### 🎞️ Posters:
- Resized to 64x64 pixels
- Normalized to [0, 1]
- TensorFlow pipelines: batching, shuffling, prefetching

### 📝 Overviews:
- Tokenized using `TextVectorization`
- Top 10,000 most frequent tokens
- Padded sequences for LSTM input

---

## 🔍 Model Architectures

### 🧩 CNN (Posters):
- 4 convolutional layers (filters: 16 → 128)
- Max pooling and Dropout for regularization
- Dense layer with 1024 units
- Sigmoid output for 25 genres
- Optimizer: Adam (1e-4), Loss: Binary Crossentropy

### ✍️ LSTM (Overviews):
- Embedding Layer (256 dimensions)
- 2 Bi-directional LSTM layers (256, 128 units)
- Dense + Dropout (ReLU + 0.5 / 0.8)
- Sigmoid output for multi-label classification

---

## 📊 Results Summary

| Metric            | CNN         | LSTM        |
|------------------|-------------|-------------|
| Validation Loss   | ~0.27       | ~0.227      |
| Precision         | 0.65        | 0.60        |
| Recall            | 0.28        | 0.23        |

- **CNN** performed well on visual patterns but struggled with rare genres
- **LSTM** captured narrative nuances better but faced challenges with class imbalance

---

## 🎯 Key Insights

- **Class imbalance** affected both models' recall
- **LSTM** was stronger in identifying narrative-driven genres
- **CNN** often misclassified nuanced genres due to visual similarity
- Future work: class weighting, attention layers, or focal loss to boost performance

---

## 🔎 Sample Prediction Analysis

| Scenario | Ground Truth | CNN Prediction | LSTM Prediction | Notes |
|---------|---------------|----------------|------------------|-------|
| Political drama | Comedy, Drama | Drama, Action, Thriller | Drama, Comedy, Romance | LSTM captured comedy nuance |
| Romantic competition | Comedy | Drama, Action, Thriller | Drama, Comedy, Romance | CNN misread visual cues |
| Suspense reunion | Horror, Mystery | Drama, Action, Thriller | Drama, Comedy, Romance | Both models struggled with suspenseful tone |

---

## 📌 Conclusion

This project demonstrates how CNN and LSTM models can be used to classify film genres using image and text data respectively. While each model showed strengths in particular areas, **dataset imbalance and model capacity** affected overall recall and genre sensitivity.

Improving this project could involve:
- **Class balancing** techniques (oversampling, weighted loss)
- **Attention mechanisms** for better sequence focus
- **Transfer learning** for image feature extraction

---

## 📁 Repository Contents

- `cnn_model.ipynb` – Poster-based CNN architecture and training
- `lstm_model.ipynb` – Overview-based LSTM model training
- `data_preprocessing.ipynb` – Poster and overview processing scripts
- `evaluation_plots/` – Model performance plots (loss, precision, recall)
- `README.md` – You are here!

---

## 📬 Contact

📧 rukayatmideibrahim@gmail.com  
🔗 [LinkedIn – Rukayat Aramide Ibrahim](https://www.linkedin.com/in/rukayat-mide-ibrahim)  

---

⭐ If you find this project useful or interesting, please give it a star and feel free to contribute or ask questions!


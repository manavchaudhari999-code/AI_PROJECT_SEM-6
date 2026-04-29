# 🧠 AI-Based RAG Medical Assistant

## 📌 Project Overview

This project is a **multimodal AI-based Medical Assistant System** that uses **Retrieval-Augmented Generation (RAG)** to generate structured clinical reports.

The system analyzes:

* Patient symptoms (text input)
* Chest X-ray images (image input)

It combines medical knowledge retrieval with image analysis to provide **accurate and explainable outputs**.

---

## 🚀 Key Features

* 🔍 Symptom-based medical knowledge retrieval using FAISS
* 🧠 Retrieval-Augmented Generation (RAG) pipeline
* 🩻 Chest X-ray analysis for pneumonia detection
* 📊 Probability-based decision fusion
* 📄 Structured clinical report generation
* 🌐 Interactive UI using Streamlit

---

## 🏗️ System Architecture

User Input (Symptoms + X-ray)
→ Text Embedding (HuggingFace)
→ FAISS Retrieval (WHO/NICE documents)
→ Image Processing (CLIP + TorchXRayVision)
→ Fusion Model (Weighted Probability)
→ LLM (LLaMA3)
→ Clinical Report Output

---

## 🧰 Technologies Used

* Python
* Streamlit
* FAISS (Vector Database)
* HuggingFace Transformers (Embeddings)
* CLIP (Image Classification)
* TorchXRayVision (X-ray Analysis)
* LLaMA3 (Report Generation)

---

## 📚 Dataset & Sources

### 🔵 Textual Data (Knowledge Base)

* WHO Clinical Guidelines
* NICE Pneumonia Guidelines

These documents are processed and stored in a FAISS vector database for retrieval.

---

### 🔵 Image Data (X-ray Input)

* NIH ChestX-ray Dataset (U.S. National Institutes of Health)
* RSNA Pneumonia Detection Dataset

These datasets are widely used in medical research and provide labeled chest X-ray images.

---

## ⚙️ Methodology

1. User inputs symptoms and uploads X-ray
2. Symptoms converted into embeddings
3. FAISS retrieves relevant medical knowledge
4. X-ray analyzed using deep learning models
5. Results combined using weighted fusion:
   **Final Score = 0.7 × X-ray + 0.3 × Symptoms**
6. LLaMA3 generates structured clinical report

---

## 📊 Evaluation

The system was tested on multiple cases including:

* High pneumonia probability
* Low probability
* Non-X-ray inputs

Based on experimental evaluation, the system achieved approximately **77% performance accuracy**, measured by correctness of diagnosis and reasoning.

---

## ⚠️ Limitations

* Limited dataset scope
* Not intended for real clinical use
* Requires computational resources

---

## 🔮 Future Scope

* Add more diseases
* Improve accuracy
* Use real-time hospital datasets
* Enhance explainability

---

## ▶️ How to Run

1. Clone the repository:

```
git clone https://github.com/manavchaudhari999-code/AI_PROJECT_SEM-6.git
```

2. Install dependencies:

```
pip install -r requirements.txt
```

3. Run the application:

```
streamlit run app.py
```

---

## 📌 Note

This project is developed for **educational and research purposes only** and should not be used for real medical diagnosis.

---

## 👨‍💻 Author

Manav Chaudhari

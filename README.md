# 🚀 AI PDF → Audiobook + RAG + Podcast System

> Convert any PDF into an **audiobook, summary, podcast, or Q&A system** using fully free AI models.

---

## 🔗 Live Demo

👉 https://huggingface.co/spaces/Faraz618/pdf-to-audiobook-ai

---

## 📸 UI Screenshot.

<img width="1350" height="602" alt="image" src="https://github.com/user-attachments/assets/73d117ee-11e2-43c8-8a7c-60652e1be9ca" />


---

## ✨ Features

* 📚 Convert PDFs into audiobooks
* 🧠 AI-powered summarization
* 🔎 RAG-based question answering
* 🎙️ Podcast-style conversation mode
* 🔊 Text-to-speech generation
* ⚡ Fully free (no paid APIs)

---

## 🧠 System Architecture

```text
PDF Upload
   ↓
Text Extraction (PyMuPDF)
   ↓
Chunking
   ↓
Embeddings (MiniLM)
   ↓
Semantic Retrieval (RAG)
   ↓
LLM Processing (Summarization / Q&A / Podcast)
   ↓
Text-to-Speech (MMS-TTS)
   ↓
Audio Merge (Pydub)
   ↓
Final Output
```

---

## 🛠️ Tech Stack

* **UI:** Gradio
* **Backend:** Python
* **PDF Parsing:** PyMuPDF
* **Embeddings:** sentence-transformers (MiniLM)
* **Similarity Search:** scikit-learn
* **LLM:** FLAN-T5 (text-generation)
* **TTS:** facebook/mms-tts
* **Audio Processing:** Pydub + FFmpeg
* **Deployment:** Hugging Face Spaces

---

## 🚀 How It Works

1. Upload a PDF
2. System extracts and processes text
3. Builds embeddings for semantic understanding
4. Retrieves relevant chunks (RAG)
5. Applies selected mode:

   * Summary
   * Podcast
   * Q&A
6. Converts output into audio
7. Returns final audiobook

---

## 🧪 Modes

### 📘 Summarized Audiobook

Condenses long documents into clean summaries and converts them into audio.

### 🎙️ Podcast Mode

Transforms content into a natural 2-speaker conversation.

### ❓ Ask Questions (RAG)

Answers questions using semantic retrieval from the document.

### 🔊 Raw Audiobook

Direct PDF → speech conversion.

---

## ⚡ Run Locally

```bash
git clone https://github.com/YOUR_USERNAME/pdf-audiobook-ai.git
cd pdf-audiobook-ai

pip install -r requirements.txt

python app.py
```

---

## 🌍 Deployment

Deployed on Hugging Face Spaces:

👉 https://huggingface.co/spaces/YOUR_USERNAME/pdf-to-audiobook-ai

---

## 🎯 Why This Project Matters

This project demonstrates:

* Real-world AI system design
* Retrieval-Augmented Generation (RAG)
* Semantic search + embeddings
* Multi-modal AI (text → audio)
* End-to-end pipeline thinking

---

## 🧠 Key Concepts

* RAG (Retrieval-Augmented Generation)
* Embeddings + similarity search
* LLM orchestration
* Audio synthesis pipelines
* AI system design

---

## 📌 Future Improvements

* Streaming audio
* Multi-voice narration
* Vector database (FAISS / Pinecone)
* Chat-based interface
* Voice cloning

---

## 👤 Author

**Faraz Mubeen Haider**

Applied AI Engineer | LLM Systems | Backend

---

## ⭐ If you like this project

Give it a star ⭐

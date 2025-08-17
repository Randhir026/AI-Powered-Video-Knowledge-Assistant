# AI-Powered-Video-Knowledge-Assistant
AI-powered YouTube Q&amp;A Assistant – Download, transcribe, search, and summarize video content using Whisper, FAISS, and Streamlit.

# 🎥 AI Video Knowledge Assistant (YouTube Q&A Bot)

An **AI-powered assistant** that helps you **ask questions about YouTube videos** and get **concise answers** with relevant transcript segments.  

The system downloads a video’s audio, transcribes it with Whisper, chunks the transcript, generates embeddings, builds a FAISS index, and enables **semantic search + summarization** inside a **Streamlit web app**.

---

## ✨ Features
- 🔊 **Download YouTube Audio** using [yt-dlp](https://github.com/yt-dlp/yt-dlp)  
- 📝 **Automatic Speech Recognition (ASR)** with [Whisper](https://huggingface.co/openai/whisper-base)  
- ✂️ **Transcript Chunking** with LangChain  
- 📊 **Semantic Embeddings** via [Sentence-Transformers](https://www.sbert.net/)  
- 🔍 **Efficient Similarity Search** using [FAISS](https://github.com/facebookresearch/faiss)  
- 🧠 **Summarization** powered by [BART-Large-CNN](https://huggingface.co/facebook/bart-large-cnn)  
- 🎨 **Interactive UI** built with [Streamlit](https://streamlit.io/)  

---

## 🏗️ Tech Stack
- **Language:** Python 3.9+  
- **Frameworks & Libraries:**  
  - Streamlit  
  - yt-dlp  
  - Hugging Face Transformers  
  - LangChain  
  - Sentence-Transformers  
  - FAISS  

---

## 🚀 Usage

1. Launch the Streamlit app in your browser.

2. Enter a YouTube video URL.

3. The system will automatically:

   - 🎧 Download & transcribe the audio

   - ✂️ Split transcript into chunks

   - 📊 Build embeddings & index with FAISS

4. Ask a question about the video.

5. ✅ Get results:

   - A concise summary answer

   - Relevant transcript segments with timestamps

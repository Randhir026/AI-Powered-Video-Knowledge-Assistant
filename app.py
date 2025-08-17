import streamlit as st
import yt_dlp
from transformers import pipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import os

# -------------------- STEP 1: Download Audio --------------------
def download_youtube_audio(url, output_filename="video_audio.mp3"):
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_filename.replace(".mp3", ".%(ext)s"),
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'quiet': False
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    return output_filename

# -------------------- STEP 2: Transcribe --------------------
@st.cache_resource
def transcribe_audio(file_path):
    asr = pipeline(
        task="automatic-speech-recognition",
        model="openai/whisper-base",  
        chunk_length_s=30,
    )
    result = asr(file_path)
    return result["text"]

# -------------------- STEP 3: Chunking --------------------
def split_into_chunks(transcript):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=20)
    chunks = text_splitter.split_text(transcript)
    metadata = [{"timestamp": f"{i*30}s"} for i in range(len(chunks))]
    return chunks, metadata

# -------------------- STEP 4: Embeddings --------------------
@st.cache_resource
def create_faiss_index(chunks):
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedder.encode(chunks, convert_to_numpy=True)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return embedder, index

# -------------------- STEP 5: Search --------------------
def search(query, embedder, index, chunks, metadata, top_k=3):
    query_embedding = embedder.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)
    results = []
    for idx, score in zip(indices[0], distances[0]):
        results.append({
            "text": chunks[idx],
            "timestamp": metadata[idx]["timestamp"],
            "score": score
        })
    return results

# -------------------- STEP 6: Summarize --------------------
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

def concise_answer(results):
    combined_text = " ".join([r["text"] for r in results])
    summarizer = load_summarizer()
    summary = summarizer(combined_text, max_length=50, min_length=25, do_sample=False)
    return summary[0]["summary_text"]

# -------------------- STREAMLIT UI --------------------
st.title("🎥 AI Video Knowledge Assistant (YouTube)")

youtube_url = st.text_input("Enter YouTube video URL:")

if youtube_url:
    with st.spinner("Downloading audio from YouTube..."):
        audio_file = download_youtube_audio(youtube_url)

    with st.spinner("Transcribing audio locally..."):
        transcript = transcribe_audio(audio_file)

    st.subheader("Transcript Preview")
    st.write(transcript[:-1] + "...")

    chunks, metadata = split_into_chunks(transcript)
    embedder, index = create_faiss_index(chunks)
    st.success(f"Transcript split into {len(chunks)} chunks and indexed.")

    query = st.text_input("Ask a question about the video:")
    if query:
        results = search(query, embedder, index, chunks, metadata)
        short_answer = concise_answer(results)

        st.subheader("📌 Short Answer")
        st.write(short_answer)

        st.subheader("Relevant Segments (Full Text)")
        for r in results:
            st.markdown(f"**[{r['timestamp']}]** {r['text']}")







import gradio as gr
import fitz
import numpy as np
from pydub import AudioSegment
from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# FIX FFMPEG
# =========================
AudioSegment.converter = "/usr/bin/ffmpeg"

# =========================
# LOAD MODELS (SAFE FOR HF)
# =========================
llm = pipeline("text-generation", model="google/flan-t5-small", max_new_tokens=256)
tts_pipeline = pipeline("text-to-speech", model="facebook/mms-tts-eng")
embedder = pipeline("feature-extraction", model="sentence-transformers/all-MiniLM-L6-v2")

# =========================
# PDF EXTRACTION
# =========================
def extract_text(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text


# =========================
# CHUNKING
# =========================
def chunk_text(text, chunk_size=150):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]


# =========================
# EMBEDDINGS
# =========================
def embed_chunks(chunks):
    embeddings = []
    for chunk in chunks:
        emb = embedder(chunk)[0]
        emb = np.mean(emb, axis=0)
        embeddings.append(emb)
    return np.array(embeddings)


# =========================
# RETRIEVAL (RAG)
# =========================
def retrieve(query, chunks, embeddings, top_k=3):
    query_emb = embedder(query)[0]
    query_emb = np.mean(query_emb, axis=0)

    sims = cosine_similarity([query_emb], embeddings)[0]
    top_indices = sims.argsort()[-top_k:][::-1]

    return [chunks[i] for i in top_indices]


# =========================
# SUMMARIZATION (LLM)
# =========================
def summarize_text(text):
    text = text[:2000]

    prompt = f"""
    Summarize this document clearly:

    {text}
    """

    result = llm(prompt)
    return result[0]["generated_text"]


# =========================
# Q&A (RAG)
# =========================
def answer_question(query, chunks, embeddings):
    relevant_chunks = retrieve(query, chunks, embeddings)
    context = " ".join(relevant_chunks)

    prompt = f"""
    Answer the question using this context:

    Context:
    {context}

    Question:
    {query}
    """

    result = llm(prompt)
    return result[0]["generated_text"]


# =========================
# PODCAST MODE
# =========================
def convert_to_podcast(text):
    text = text[:1500]
    sentences = text.split(". ")

    convo = ""
    for i, s in enumerate(sentences):
        speaker = "Host" if i % 2 == 0 else "Guest"
        convo += f"{speaker}: {s.strip()}.\n"

    return convo


# =========================
# TTS
# =========================
def generate_audio(text, filename):
    output = tts_pipeline(text)

    with open(filename, "wb") as f:
        f.write(output["audio"])

    return filename


# =========================
# MERGE AUDIO
# =========================
def merge_audio(files, output_path):
    combined = AudioSegment.empty()
    for f in files:
        combined += AudioSegment.from_file(f)
    combined.export(output_path, format="mp3")
    return output_path


# =========================
# MAIN PIPELINE
# =========================
def process(pdf, mode, question, chunk_size, progress=gr.Progress()):
    if pdf is None:
        raise gr.Error("Upload PDF")

    progress(0, desc="Reading PDF...")
    text = extract_text(pdf)

    progress(0.2, desc="Chunking...")
    chunks = chunk_text(text, chunk_size)

    progress(0.3, desc="Embedding...")
    embeddings = embed_chunks(chunks)

    # ------------------------
    # MODES
    # ------------------------
    if mode == "Summarized Audiobook":
        processed_text = summarize_text(text)

    elif mode == "Podcast Mode":
        processed_text = convert_to_podcast(text)

    elif mode == "Ask Questions":
        if not question:
            raise gr.Error("Enter question")
        processed_text = answer_question(question, chunks, embeddings)

    else:
        processed_text = text[:2000]

    # ------------------------
    # AUDIO
    # ------------------------
    progress(0.5, desc="Generating audio...")
    audio_files = []
    text_chunks = chunk_text(processed_text, chunk_size)

    for i, chunk in enumerate(text_chunks):
        progress(0.5 + (0.4 * (i / len(text_chunks))), desc=f"Audio {i+1}/{len(text_chunks)}")
        filename = f"chunk_{i}.wav"
        generate_audio(chunk, filename)
        audio_files.append(filename)

    progress(0.95, desc="Merging...")
    final_audio = "final.mp3"
    merge_audio(audio_files, final_audio)

    progress(1.0, desc="Done")

    return final_audio, processed_text


# =========================
# UI
# =========================
with gr.Blocks() as demo:
    gr.Markdown("# 🚀 AI PDF → Audiobook + RAG + Podcast (FREE)")

    pdf = gr.File(label="Upload PDF")

    mode = gr.Radio(
        ["Raw Audiobook", "Summarized Audiobook", "Podcast Mode", "Ask Questions"],
        value="Summarized Audiobook",
        label="Mode"
    )

    question = gr.Textbox(label="Ask Question (for Q&A mode)")

    chunk_size = gr.Slider(50, 300, value=150, label="Chunk Size")

    btn = gr.Button("Run AI")

    audio = gr.Audio(label="Audio Output")
    text_out = gr.Textbox(label="AI Output", lines=15)

    btn.click(
        fn=process,
        inputs=[pdf, mode, question, chunk_size],
        outputs=[audio, text_out]
    )

demo.launch()

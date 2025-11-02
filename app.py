import os
import shutil
import time
import streamlit as st
import pdfplumber
import tempfile
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from huggingface_hub import InferenceClient

from langchain_core.prompts import ChatPromptTemplate

# -----------------------------
# 🔧 Load environment variables
# -----------------------------
load_dotenv()

HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "HuggingFaceH4/zephyr-7b-beta")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHROMA_DB_DIR = os.getenv("CHROMA_DB_DIR", "chroma_langchain_hf")

if HF_TOKEN:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN

# Clear old Chroma cache (optional)
if os.path.exists(CHROMA_DB_DIR):
    for i in range(5):
        try:
            shutil.rmtree(CHROMA_DB_DIR)
            break
        except PermissionError:
            time.sleep(1)

# -----------------------------
# ⚙️ Streamlit setup
# -----------------------------
st.set_page_config(page_title="📄 Explore Your PDF", page_icon="📚")
st.title("📄 Explore Your PDF — Hugging Face Edition")
st.caption("Upload a PDF and ask questions about its content using a Hugging Face model.")

# -----------------------------
# 🤗 Model selection (sidebar)
# -----------------------------
st.sidebar.header("⚙️ Model Settings")
model_name = st.sidebar.text_input("Model name (repo ID)", value=DEFAULT_MODEL)

if not HF_TOKEN:
    st.warning("No Hugging Face token found. Add it to your .env file.")
    st.stop()

# -----------------------------
# 📥 Upload PDF
# -----------------------------
uploaded_file = st.file_uploader("📎 Upload a PDF file", type=["pdf"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        pdf_path = tmp.name

    st.success("✅ PDF uploaded successfully!")

    # -----------------------------
    # 📄 Extract text
    # -----------------------------
    all_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                all_text += text + "\n"

    if not all_text.strip():
        st.error("❌ Could not extract text from this PDF.")
        st.stop()

    # -----------------------------
    # ✂️ Chunk text
    # -----------------------------
    st.info("🔍 Splitting text into chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    texts = splitter.split_text(all_text)
    st.write(f"📚 Created {len(texts)} chunks.")

    # -----------------------------
    # 💾 Create / load Chroma store
    # -----------------------------
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    if os.path.exists(CHROMA_DB_DIR) and os.listdir(CHROMA_DB_DIR):
        vectordb = Chroma(
            persist_directory=CHROMA_DB_DIR,
            embedding_function=embeddings
        )
    else:
        vectordb = Chroma.from_texts(
            texts=texts,
            embedding=embeddings,
            persist_directory=CHROMA_DB_DIR
        )
    st.success("Chunking and embedding completed successfully!")

    retriever = vectordb.as_retriever(search_kwargs={"k": 3})

    # -----------------------------
    # 🧠 Hugging Face Client (LLM)
    # -----------------------------
    HF_MODEL_ID = model_name
    client = InferenceClient(model=HF_MODEL_ID, token=HF_TOKEN)

    def call_llm(prompt):
        response = client.chat_completion(
            model=HF_MODEL_ID,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
        )
        return response.choices[0].message["content"]

    # -----------------------------
    # 💬 Ask a question
    # -----------------------------
    st.success("You can now ask your question:")
    query = st.text_input("Ask a question about your PDF:")

    if query:
        with st.spinner("🤖 Thinking..."):
            docs = retriever.invoke(query)
            context = "\n\n".join([d.page_content for d in docs])

            prompt_template = """Answer the user's question based on the provided context.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    Context:
    {context}

    Question:
    {question}"""

            prompt = prompt_template.format(context=context, question=query)
            answer = call_llm(prompt)

        st.subheader("💡 Answer")
        st.write(answer)

        with st.expander("📘 Retrieved Context"):
            st.write(context)


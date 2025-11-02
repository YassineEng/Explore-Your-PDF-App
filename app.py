import os
import shutil
import time
import streamlit as st
import pdfplumber
import tempfile
from dotenv import load_dotenv
import hashlib
import atexit
import gc

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from huggingface_hub import InferenceClient

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
    file_content = uploaded_file.read()
    file_hash = hashlib.md5(file_content).hexdigest()
    unique_chroma_db_dir = os.path.join(CHROMA_DB_DIR, file_hash)

    if "last_uploaded_file_hash" not in st.session_state or st.session_state.last_uploaded_file_hash != file_hash:
        st.session_state.last_uploaded_file_hash = file_hash
        st.session_state.vectordb = None  # Clear previous vectordb

        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(file_content)
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

        st.session_state.vectordb = Chroma.from_texts(
            texts=texts,
            embedding=embeddings,
            persist_directory=unique_chroma_db_dir
        )
        st.success("Chunking and embedding completed successfully!")

    # Load existing Chroma store if the file is the same or if it was just created
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    if st.session_state.vectordb is None:
        st.session_state.vectordb = Chroma(
            persist_directory=unique_chroma_db_dir,
            embedding_function=embeddings
        )
    retriever = st.session_state.vectordb.as_retriever(search_kwargs={"k": 3})

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

            prompt_template = """Answer the user's question truthfully and *only* based on the provided context. Do NOT use any external knowledge.
    If the answer is not found *within the provided context*, respond with "I cannot answer this question based on the provided document."

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

# Cleanup function to delete all Chroma subdirectories on app shutdown
def cleanup_chroma_db():
    print(f"Attempting to clean up Chroma DB directory: {CHROMA_DB_DIR}")
    # Add a small delay to allow file handles to be released
    time.sleep(0.5)

    if "vectordb" in st.session_state and st.session_state.vectordb is not None:
        try:
            # Attempt to close the underlying Chroma client if it has a close method
            if hasattr(st.session_state.vectordb._client, 'close'):
                st.session_state.vectordb._client.close()
                print("  Chroma client explicitly closed.")
            st.session_state.vectordb = None
            gc.collect()
            time.sleep(0.5) # Additional delay after closing
        except Exception as e:
            print(f"  Error closing Chroma client: {e}")

    if os.path.exists(CHROMA_DB_DIR):
        for item in os.listdir(CHROMA_DB_DIR):
            item_path = os.path.join(CHROMA_DB_DIR, item)
            if os.path.isdir(item_path):
                print(f"  Deleting subdirectory: {item_path}")
                for i in range(5):
                    try:
                        shutil.rmtree(item_path)
                        print(f"    Successfully deleted: {item_path}")
                        break
                    except PermissionError as e:
                        print(f"    PermissionError deleting {item_path}: {e}. Retrying in 1 second...")
                        time.sleep(1)
                else:
                    print(f"    Failed to delete {item_path} after multiple retries due to PermissionError.")
    print("Chroma DB cleanup attempt finished.")

atexit.register(cleanup_chroma_db)

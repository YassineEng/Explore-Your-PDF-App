# 📄 Explore Your PDF - Hugging Face Edition

This Streamlit web application allows you to have a conversation with your PDF documents. Upload a PDF, and the app will use a Hugging Face language model to answer your questions about its content.

## ✨ Features

-   **📄 PDF Upload**: Upload any PDF file directly through the web interface.
-   **💬 Question Answering**: Ask questions in natural language about the content of your PDF.
-   **🤗 Hugging Face Integration**: Utilizes Hugging Face models for question answering and embeddings.
-   **⚙️ Customizable Model**: Choose the Hugging Face model you want to use for question answering.
-   **🧠 Smart Context Retrieval**: The application intelligently retrieves relevant parts of the PDF to answer your questions.
-   **🗑️ Automatic Cleanup**: The application automatically cleans up the created vector store on exit.

## 🚀 Getting Started

### Prerequisites

-   Python 3.7+
-   A Hugging Face API token

### Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/Explore-Your-PDF-App.git
    cd Explore-Your-PDF-App
    ```

2.  **Create a virtual environment and activate it:**

    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
    ```

3.  **Install the dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up your environment variables:**

    Create a `.env` file in the root of the project and add your Hugging Face API token:

    ```
    HUGGINGFACEHUB_API_TOKEN="your-hugging-face-api-token"
    ```

    You can also customize the models used by adding the following variables:

    ```
    DEFAULT_MODEL="HuggingFaceH4/zephyr-7b-beta"
    EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2"
    ```

### Running the Application

To start the Streamlit application, run the following command in your terminal:

```bash
streamlit run app.py
```

The application will open in your web browser.

## 🛠️ Technologies Used

-   **Streamlit**: For the web application framework.
-   **LangChain**: For text splitting, embeddings, and vector store management.
-   **Hugging Face**: For language models and embeddings.
-   **Chroma**: As the vector store for storing and retrieving text embeddings.
-   **pdfplumber**: For extracting text from PDF files.

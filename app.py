import os
import streamlit as st
from dotenv import load_dotenv

from src.utils import ensure_directories_exist, UPLOADS_DIR, clear_knowledge_base, format_citations
from src.rag_pipeline import RAGPipeline
from src.pdf_loader import PDFProcessingError
from src.vector_store import VectorStoreManager
from src.llm import LLMError

# Load environment variables
load_dotenv()

# Page Configuration
st.set_page_config(
    page_title="PDF RAG Assistant",
    page_icon="📄",
    layout="wide"
)

# Ensure runtime directories exist
ensure_directories_exist()


# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

if "groq_api_key" not in st.session_state:
    st.session_state.groq_api_key = os.getenv("GROQ_API_KEY", "")

if "gemini_api_key" not in st.session_state:
    st.session_state.gemini_api_key = os.getenv("GEMINI_API_KEY", os.getenv("GOOGLE_API_KEY", ""))

if "openai_api_key" not in st.session_state:
    st.session_state.openai_api_key = os.getenv("OPENAI_API_KEY", "")


# Sidebar Configuration
with st.sidebar:
    st.title("⚙️ PDF RAG Settings")

    # LLM Provider Selection
    provider_option = st.selectbox(
        "LLM Provider",
        options=["Groq (Instant Free)", "Google Gemini (Free)", "OpenAI"],
        index=0,
        help="Groq & Gemini offer 100% FREE API keys without credit card or billing!"
    )

    if "Groq" in provider_option:
        provider_code = "groq"
        if not st.session_state.groq_api_key:
            st.session_state.groq_api_key = os.getenv("GROQ_API_KEY", "")

        groq_input = st.text_input(
            "Groq API Key",
            value=st.session_state.groq_api_key,
            type="password",
            help="Enter your free Groq API key."
        )
        if groq_input:
            st.session_state.groq_api_key = groq_input

        if st.session_state.groq_api_key:
            st.caption("✅ Groq API Key is active (Loaded from `.env`)")
        else:
            st.markdown(
                "👉 **[Get Instant FREE Groq API Key](https://console.groq.com/keys)** *(1-click setup, no card needed)*"
            )
        active_api_key = st.session_state.groq_api_key

    elif "Gemini" in provider_option:
        provider_code = "gemini"
        gemini_input = st.text_input(
            "Google Gemini API Key",
            value=st.session_state.gemini_api_key,
            type="password",
            help="Enter your free Google Gemini API key."
        )
        if gemini_input:
            st.session_state.gemini_api_key = gemini_input

        st.markdown(
            "👉 **[Get FREE Gemini API Key](https://aistudio.google.com/app/apikey)**"
        )
        active_api_key = st.session_state.gemini_api_key

    else:
        provider_code = "openai"
        openai_input = st.text_input(
            "OpenAI API Key",
            value=st.session_state.openai_api_key,
            type="password",
            help="Enter your OpenAI API key."
        )
        if openai_input:
            st.session_state.openai_api_key = openai_input
        active_api_key = st.session_state.openai_api_key

    st.divider()

    # RAG Hyperparameters
    st.subheader("Chunking & Retrieval")
    chunk_size = st.slider("Chunk Size", min_value=100, max_value=2000, value=500, step=50,
                           help="Number of characters per text chunk.")
    chunk_overlap = st.slider("Chunk Overlap", min_value=0, max_value=500, value=50, step=10,
                             help="Overlap characters between adjacent chunks.")
    top_k = st.slider("Top K Chunks", min_value=1, max_value=10, value=3, step=1,
                      help="Number of relevant chunks retrieved for answering.")

    st.divider()

    # File Upload Section
    st.subheader("Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload one or more PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

    col1, col2 = st.columns(2)

    with col1:
        build_btn = st.button("🔨 Build Index", use_container_width=True, type="primary")

    with col2:
        clear_btn = st.button("🗑️ Clear Index", use_container_width=True)

    # Action: Clear Knowledge Base
    if clear_btn:
        clear_knowledge_base()
        st.session_state.messages = []
        st.success("Knowledge base and chat history cleared!")
        st.rerun()

    # Knowledge Base Status indicator
    st.divider()
    if VectorStoreManager.vector_store_exists():
        st.success("✅ Knowledge base ready!")
    else:
        st.info("ℹ️ No active index. Upload PDFs and build index.")


# Action: Build Knowledge Base
if build_btn:
    if not uploaded_files:
        st.sidebar.error("Please select at least one PDF file to upload.")
    else:
        with st.spinner("Processing PDFs, chunking text, and building FAISS index..."):
            saved_paths = []
            try:
                # Save uploaded files to data/uploads
                for file in uploaded_files:
                    path = os.path.join(UPLOADS_DIR, file.name)
                    with open(path, "wb") as f:
                        f.write(file.getbuffer())
                    saved_paths.append(path)

                # Initialize pipeline and build index
                pipeline = RAGPipeline(api_key=active_api_key, provider=provider_code)
                stats = pipeline.build_knowledge_base(
                    file_paths=saved_paths,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )

                st.sidebar.success(
                    f"Index created!\n\n"
                    f"• PDFs: {stats['num_documents']}\n"
                    f"• Pages: {stats['num_pages']}\n"
                    f"• Chunks: {stats['num_chunks']}"
                )
                st.rerun()

            except PDFProcessingError as e:
                st.sidebar.error(f"PDF Error: {str(e)}")
            except Exception as e:
                st.sidebar.error(f"Error building index: {str(e)}")


# Main Interface
st.title("📄 PDF RAG Assistant")
st.caption(
    f"Powered by FAISS Vector Retrieval and **{provider_option}**. "
    "Upload your PDFs in the sidebar, build the index, and ask questions about your documents."
)

# Display Chat Messages from Session State
# Display Chat Messages from Session State
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "citations" in msg and msg["citations"]:
            st.markdown("**Sources:**")
            for cit in msg["citations"]:
                st.markdown(cit)
        if "images" in msg and msg["images"]:
            with st.expander("🖼️ Visual Page Context / Diagrams", expanded=False):
                cols = st.columns(min(len(msg["images"]), 3))
                for idx, img_info in enumerate(msg["images"]):
                    img_p = img_info.get("path")
                    if img_p and os.path.exists(img_p):
                        with cols[idx % len(cols)]:
                            st.image(img_p, caption=f"Source: {img_info.get('source')} (Page {img_info.get('page')})", use_container_width=True)
        if "context_chunks" in msg and msg["context_chunks"]:
            with st.expander("▼ Retrieved Context"):
                for idx, chunk in enumerate(msg["context_chunks"], 1):
                    st.markdown(f"**Chunk {idx}** | Source: `{chunk['source']}` | Page: `{chunk['page_number']}` | Score: `{chunk['score']}`")
                    st.text(chunk['content'])
                    st.divider()


# Question Input
if user_question := st.chat_input("Ask a question about your uploaded documents..."):

    # Append user question
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    # Generate Response
    with st.chat_message("assistant"):
        if not VectorStoreManager.vector_store_exists():
            error_msg = "Please upload a PDF and click 'Build Index' in the sidebar first."
            st.warning(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
        else:
            with st.spinner("Searching document context and generating answer..."):
                try:
                    pipeline = RAGPipeline(api_key=active_api_key, provider=provider_code)
                    result = pipeline.answer_question(question=user_question, top_k=top_k)

                    answer = result["answer"]
                    sources = result["sources"]
                    context_chunks = result["context_chunks"]
                    citations = format_citations(sources)

                    # Collect unique images associated with retrieved chunks
                    retrieved_images = []
                    seen_imgs = set()
                    for chunk in context_chunks:
                        img_p = chunk.get("image_path")
                        if img_p and os.path.exists(img_p) and img_p not in seen_imgs:
                            seen_imgs.add(img_p)
                            retrieved_images.append({
                                "path": img_p,
                                "source": chunk.get("source"),
                                "page": chunk.get("page_number")
                            })

                    # Display Answer
                    st.markdown(answer)

                    # Display Sources
                    if citations:
                        st.markdown("**Sources:**")
                        for cit in citations:
                            st.markdown(cit)

                    # Display Retrieved Images / Diagrams
                    if retrieved_images:
                        with st.expander("🖼️ Visual Page Context / Diagrams", expanded=True):
                            cols = st.columns(min(len(retrieved_images), 3))
                            for idx, img_info in enumerate(retrieved_images):
                                with cols[idx % len(cols)]:
                                    st.image(
                                        img_info["path"],
                                        caption=f"Source: {img_info['source']} (Page {img_info['page']})",
                                        use_container_width=True
                                    )

                    # Display Retrieved Context Expander
                    if context_chunks:
                        with st.expander("▼ Retrieved Context"):
                            for idx, chunk in enumerate(context_chunks, 1):
                                st.markdown(f"**Chunk {idx}** | Source: `{chunk['source']}` | Page: `{chunk['page_number']}` | Distance Score: `{chunk['score']}`")
                                st.text(chunk['content'])
                                st.divider()

                    # Save response to session state
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "citations": citations,
                        "images": retrieved_images,
                        "context_chunks": context_chunks
                    })

                except LLMError as e:
                    err_txt = f"⚠️ {str(e)}"
                    st.error(err_txt)
                    st.session_state.messages.append({"role": "assistant", "content": err_txt})
                except Exception as e:
                    err_txt = f"⚠️ Error generating answer: {str(e)}"
                    st.error(err_txt)
                    st.session_state.messages.append({"role": "assistant", "content": err_txt})


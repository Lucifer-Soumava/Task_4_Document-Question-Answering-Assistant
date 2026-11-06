from pathlib import Path
import streamlit as st

from src.document_loader import load_document, load_folder
from src.qa_engine import DocumentQA, FALLBACK_MESSAGE

BASE_DIR = Path(__file__).resolve().parent
SAMPLE_DIR = BASE_DIR / "data" / "sample_corpus"

st.set_page_config(page_title="Document Q&A Assistant", page_icon="📚", layout="wide")
st.title("Document Question-Answering Assistant")
st.caption(
    "Ask questions about your documents. Answers are extracted from retrieved "
    "passages and include source citations. Everything runs locally."
)

with st.sidebar:
    st.header("Document library")
    use_samples = st.checkbox("Include sample documents", value=True)
    uploaded_files = st.file_uploader(
        "Add your documents", type=["pdf", "docx", "txt", "md"], accept_multiple_files=True,
        help="PDF, DOCX, TXT, and Markdown are supported. Scanned image-only PDFs need OCR first.",
    )
    st.divider()
    st.markdown("**Retrieval method**")
    st.write("TF-IDF (word unigrams and bigrams) with extractive sentence matching.")
    st.caption("No paid API, hosted model, or external network call is used.")

segments = []
loaded_names = []
if use_samples:
    sample_segments = load_folder(SAMPLE_DIR)
    segments.extend(sample_segments)
    loaded_names.extend({item["source"] for item in sample_segments})

if uploaded_files:
    for uploaded in uploaded_files:
        try:
            extracted = load_document(uploaded.name, uploaded.getvalue())
            segments.extend(extracted)
            loaded_names.append(uploaded.name)
        except Exception as exc:
            st.error(f"Could not load {uploaded.name}: {exc}")

if segments:
    qa = DocumentQA.from_segments(segments)
    st.success(f"Loaded {len(set(loaded_names))} document(s) and indexed {len(qa.chunks)} passage chunk(s).")
    with st.expander("View loaded documents"):
        for name in sorted(set(loaded_names)):
            st.write(f"• {name}")

    st.subheader("Ask a question")
    with st.form("question_form"):
        question = st.text_input(
            "Your question",
            placeholder="Example: How long may an undergraduate borrow a book?",
        )
        submitted = st.form_submit_button("Find answer", type="primary")

    if submitted:
        if not question.strip():
            st.warning("Enter a question first.")
        else:
            result = qa.answer(question)
            st.markdown("### Answer")
            if result["supported"]:
                st.markdown(result["answer"])
                st.caption(f"Retrieval relevance (heuristic): {result['relevance']:.3f}")
                st.markdown("### Supporting source passages")
                for passage in result["passages"]:
                    with st.expander(passage["citation"], expanded=True):
                        st.write(passage["text"])
            else:
                st.warning(FALLBACK_MESSAGE)
                st.caption("No sufficiently relevant supporting passage was found in the loaded documents.")
else:
    st.info("Upload one or more documents, or enable the sample documents in the sidebar.")

st.divider()
st.caption(
    "Grounding policy: answers are selected from source sentences, not generated as free-form facts. "
    "Each returned sentence is linked to a supporting passage."
)

import os

import chromadb
import streamlit as st
from chromadb.config import Settings
from google import genai
from sentence_transformers import SentenceTransformer


st.set_page_config(
    page_title="College Knowledge Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

    :root {
        --ink: #f5f7ff;
        --muted: #9ba7c5;
        --line: rgba(155, 172, 221, 0.18);
        --panel: rgba(15, 24, 52, 0.72);
        --blue: #65b8ff;
        --violet: #9b8cff;
    }

    .stApp {
        color: var(--ink);
        background:
            radial-gradient(circle at 82% 4%, rgba(82, 108, 255, 0.18), transparent 28rem),
            radial-gradient(circle at 12% 24%, rgba(0, 173, 255, 0.10), transparent 25rem),
            linear-gradient(135deg, #050916 0%, #091126 52%, #070b19 100%);
        font-family: 'DM Sans', sans-serif;
    }

    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stToolbar"] { visibility: hidden; }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(7, 13, 30, 0.98), rgba(10, 16, 36, 0.94));
        border-right: 1px solid var(--line);
    }
    [data-testid="stSidebarContent"] { padding: 2rem 1.35rem; }
    [data-testid="stSidebar"] h3 {
        color: #dfe7ff;
        font-family: 'Space Grotesk', sans-serif;
        letter-spacing: .08em;
        font-size: .72rem;
        text-transform: uppercase;
        margin-top: 1.7rem;
    }
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] li {
        color: var(--muted);
        font-size: .88rem;
        line-height: 1.65;
    }
    [data-testid="stSidebar"] ul { padding-left: 1.15rem; }

    .block-container { max-width: 1120px; padding: 3.2rem 2rem 2rem; }
    .hero {
        position: relative;
        overflow: hidden;
        padding: 3.7rem 3.5rem 3.5rem;
        border: 1px solid rgba(132, 165, 255, .22);
        border-radius: 28px;
        background: linear-gradient(135deg, rgba(17, 30, 68, .88), rgba(11, 17, 39, .62));
        box-shadow: 0 26px 80px rgba(0, 0, 0, .32), inset 0 1px rgba(255,255,255,.06);
    }
    .hero::after {
        content: '';
        position: absolute;
        width: 330px;
        height: 330px;
        right: -100px;
        top: -160px;
        border-radius: 50%;
        background: rgba(95, 155, 255, .15);
        filter: blur(35px);
    }
    .eyebrow {
        color: #9ebfff;
        font-size: .74rem;
        font-weight: 700;
        letter-spacing: .18em;
        text-transform: uppercase;
    }
    .hero h1 {
        position: relative;
        z-index: 1;
        margin: .7rem 0 .65rem;
        color: var(--ink);
        font-family: 'Space Grotesk', sans-serif;
        font-size: clamp(2.2rem, 5vw, 4.5rem);
        line-height: 1.03;
        letter-spacing: -.04em;
    }
    .hero .tagline {
        position: relative;
        z-index: 1;
        margin: 0;
        color: #dce6ff;
        font-family: 'Space Grotesk', sans-serif;
        font-size: clamp(1.15rem, 2vw, 1.55rem);
    }
    .hero .supporting {
        position: relative;
        z-index: 1;
        max-width: 570px;
        margin: 1.25rem 0 0;
        color: var(--muted);
        font-size: 1rem;
        line-height: 1.7;
    }
    .badge {
        position: relative;
        z-index: 1;
        display: inline-block;
        margin-top: 2rem;
        padding: .55rem .85rem;
        border: 1px solid rgba(117, 184, 255, .28);
        border-radius: 999px;
        background: rgba(95, 140, 255, .09);
        color: #b9d5ff;
        font-size: .7rem;
        font-weight: 700;
        letter-spacing: .12em;
    }
    .section-label {
        margin: 2.8rem 0 .9rem;
        color: #dce5ff;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.25rem;
        font-weight: 600;
    }
    .question-shell {
        padding: 1.4rem;
        border: 1px solid var(--line);
        border-radius: 20px;
        background: rgba(12, 20, 44, .62);
        box-shadow: 0 20px 60px rgba(0, 0, 0, .2);
    }
    [data-testid="stTextArea"] textarea {
        min-height: 120px;
        border: 1px solid rgba(140, 164, 222, .25) !important;
        border-radius: 14px !important;
        background: rgba(4, 10, 25, .62) !important;
        color: #f4f7ff !important;
        font-size: 1.05rem !important;
        line-height: 1.55 !important;
        padding: 1rem !important;
    }
    [data-testid="stTextArea"] textarea:focus {
        border-color: rgba(101, 184, 255, .75) !important;
        box-shadow: 0 0 0 1px rgba(101, 184, 255, .25), 0 0 28px rgba(72, 135, 255, .15) !important;
    }
    .stButton > button {
        width: 100%;
        min-height: 3.05rem;
        border: 1px solid rgba(128, 189, 255, .42);
        border-radius: 12px;
        background: linear-gradient(100deg, #317ed9, #705cd6);
        color: white;
        font-family: 'Space Grotesk', sans-serif;
        font-size: .95rem;
        font-weight: 700;
        box-shadow: 0 10px 30px rgba(60, 110, 229, .25);
        transition: transform .2s ease, box-shadow .2s ease;
    }
    .stButton > button:hover {
        border-color: rgba(173, 218, 255, .8);
        box-shadow: 0 12px 34px rgba(75, 126, 244, .42);
        transform: translateY(-1px);
    }
    .answer-card, .source-card {
        border: 1px solid var(--line);
        border-radius: 18px;
        background: linear-gradient(135deg, rgba(19, 31, 66, .82), rgba(12, 18, 39, .76));
        box-shadow: 0 20px 55px rgba(0, 0, 0, .22), inset 0 1px rgba(255,255,255,.035);
    }
    .answer-card { padding: 1.75rem 1.9rem; margin-top: 1.25rem; }
    .card-kicker {
        color: #9bc8ff;
        font-size: .72rem;
        font-weight: 700;
        letter-spacing: .14em;
    }
    .answer-copy {
        margin-top: 1rem;
        color: #f0f4ff;
        font-size: 1.06rem;
        line-height: 1.8;
    }
    .metadata {
        margin-top: 1.25rem;
        color: #8291b4;
        font-size: .78rem;
    }
    [data-testid="stExpander"] {
        margin-top: 1rem;
        border: 1px solid var(--line) !important;
        border-radius: 16px !important;
        background: rgba(10, 17, 38, .52) !important;
    }
    [data-testid="stExpander"] summary p {
        color: #dce5ff;
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 600;
    }
    .source-card { padding: 1.15rem 1.3rem; margin: .75rem 0; }
    .source-label { color: #8ebaff; font-size: .7rem; font-weight: 700; letter-spacing: .13em; }
    .source-text { margin-top: .7rem; color: #bcc8e3; font-size: .9rem; line-height: 1.65; white-space: pre-wrap; }
    .footer { margin: 3.5rem 0 1rem; color: #6f7d9f; font-size: .78rem; letter-spacing: .08em; text-align: center; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_rag_resources():
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    chroma_client = chromadb.PersistentClient(
        path="chroma_rag_db",
        settings=Settings(chroma_api_impl="chromadb.api.segment.SegmentAPI"),
    )
    collection = chroma_client.get_collection(
        name="college_policies",
        embedding_function=None,
    )
    gemini_client = genai.Client(api_key=os.environ["Gemini_API_Key"])
    return embedding_model, collection, gemini_client


def answer_question(question):
    embedding_model, collection, gemini_client = load_rag_resources()
    question_embedding = embedding_model.encode(
        question,
        convert_to_numpy=True,
    ).tolist()
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=3,
        include=["documents", "distances"],
    )
    retrieved_chunks = results["documents"][0]
    context = "\n\n".join(
        f"Chunk {index}:\n{chunk}"
        for index, chunk in enumerate(retrieved_chunks, start=1)
    )
    prompt = f"""Answer the question using only the provided context.
If the context does not contain the answer, say that the answer is not
available in the retrieved context. Do not use general knowledge.

Question:
{question}

Retrieved context:
{context}
"""
    response = gemini_client.models.generate_content(
        model="gemini-flash-lite-latest",
        contents=prompt,
    )
    return response.text, retrieved_chunks


with st.sidebar:
    st.markdown("## 🎓 College AI")
    st.markdown("### About")
    st.markdown(
        "College Knowledge Assistant uses Retrieval-Augmented Generation to "
        "retrieve relevant information from college documents before generating an answer."
    )
    st.markdown("### Tech Stack")
    st.markdown("- Python\n- PyPDF\n- Sentence Transformers\n- ChromaDB\n- Gemini\n- Streamlit")
    st.markdown("### Sample Questions")
    sample_questions = [
        "What is the attendance policy?",
        "What are the placement requirements?",
        "What are the student conduct rules?",
    ]
    for sample in sample_questions:
        if st.button(sample, key=f"sample_{sample}"):
            st.session_state.question = sample


st.markdown(
    """
    <section class="hero">
        <div class="eyebrow">🎓 College Knowledge Assistant</div>
        <h1>Ask your college.<br>Get intelligent answers.</h1>
        <p class="tagline">A smarter way to navigate the policies that matter.</p>
        <p class="supporting">An AI-powered RAG assistant that understands your college policy documents.</p>
        <div class="badge">✦ RAG POWERED &nbsp;•&nbsp; DOCUMENT GROUNDED</div>
    </section>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="section-label">What would you like to know?</div>', unsafe_allow_html=True)

if "question" not in st.session_state:
    st.session_state.question = ""

st.markdown('<div class="question-shell">', unsafe_allow_html=True)
question = st.text_area(
    "Question",
    value=st.session_state.question,
    placeholder="Ask something about attendance, placements, student policies...",
    label_visibility="collapsed",
    key="question_input",
)
ask_clicked = st.button("✦ Ask Assistant", type="primary")
st.markdown("</div>", unsafe_allow_html=True)

if ask_clicked:
    if not question.strip():
        st.warning("Enter a question to begin.")
    else:
        with st.spinner("Searching your college policy documents..."):
            try:
                answer, sources = answer_question(question.strip())
                st.session_state.answer = answer
                st.session_state.sources = sources
            except Exception as error:
                st.error(f"Unable to generate an answer: {error}")

if "answer" in st.session_state:
    st.markdown(
        f'<div class="answer-card"><div class="card-kicker">✦ AI ANSWER</div>'
        f'<div class="answer-copy">{st.session_state.answer}</div>'
        '<div class="metadata">Grounded in your college policy document</div></div>',
        unsafe_allow_html=True,
    )
    with st.expander("⌄ View Retrieved Sources", expanded=False):
        for index, source in enumerate(st.session_state.sources, start=1):
            st.markdown(
                f'<div class="source-card"><div class="source-label">SOURCE {index:02d}</div>'
                f'<div class="source-text">{source}</div></div>',
                unsafe_allow_html=True,
            )

st.markdown('<div class="footer">Built with RAG &nbsp;•&nbsp; Powered by AI</div>', unsafe_allow_html=True)

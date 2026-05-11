import streamlit as st
from utils import write_message
from agent import generate_response

st.set_page_config(
    page_title="Nigerian Student Rights QA",
    page_icon="⚖️",
    layout="wide",
)

# ── Sidebar ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚖️ Nigerian Student Rights")
    st.markdown(
        "An intelligent QA system powered by a **Neo4j Knowledge Graph** "
        "and **OpenAI** language models."
    )
    st.divider()

    st.subheader("Key Topics")
    st.markdown(
        """
        - SERVICOM policies & obligations
        - Student admission rights
        - Examination & grading rights
        - Disciplinary procedures
        - Grievance & complaint channels
        - Welfare & accommodation rights
        - Academic freedom
        - NUC regulations
        """
    )
    st.divider()

    st.subheader("How to use")
    st.markdown(
        """
        1. Type your question in the chat box below.
        2. The system searches the **SERVICOM and Student Rights** knowledge base.
        3. Answers are grounded in the actual documents — not guesswork.
        """
    )
    st.divider()

    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Chat cleared. How can I help you with Nigerian student rights?"
                ),
            }
        ]
        st.rerun()

    st.caption("Powered by Neo4j · LangChain · OpenAI")

# ── Session state ─────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hello! I am the **Nigerian Student Rights QA Assistant**. "
                "I can help you understand your rights as a student in Nigerian "
                "tertiary institutions — including SERVICOM policies, NUC regulations, "
                "university obligations, grievance procedures, and more.\n\n"
                "What would you like to know?"
            ),
        }
    ]

# ── Main header ───────────────────────────────────────────────────────────
st.title("⚖️ Nigerian Student Rights QA System")
st.caption(
    "Ask any question about student rights, SERVICOM policies, "
    "and university regulations in Nigeria."
)

# ── Display conversation ──────────────────────────────────────────────────
for message in st.session_state.messages:
    write_message(message["role"], message["content"], save=False)

# ── Handle input ──────────────────────────────────────────────────────────
if question := st.chat_input("Ask about your student rights..."):
    write_message("user", question)

    with st.spinner("Searching knowledge base..."):
        response = generate_response(question)
        write_message("assistant", response)
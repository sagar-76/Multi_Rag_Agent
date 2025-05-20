
import streamlit as st
from ML_rag import PDFRAGQA
from m import SQLAgentGroq

# === Initialize PDF RAG ===
@st.cache_resource(show_spinner=False)
def init_pdf_rag():
    rag = PDFRAGQA(groq_api_key="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")
    text = rag.load_pdf("D:\\download\\sagar\\The Hundred-Page Machine Learning Book PDF.pdf")
    rag.create_vector_store(text)
    return rag

# === Initialize SQL Agent ===
@st.cache_resource(show_spinner=False)
def init_sql_agent():
    sql_agent = SQLAgentGroq(
        groq_api_key="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",
        db_uri="mysql+pymysql://root:root@127.0.0.1:3306/campusx"
    )
    return sql_agent


st.title("PDF RAG & SQL Agent (Direct)")

rag = init_pdf_rag()
sql_agent = init_sql_agent()

tab1, tab2 = st.tabs(["PDF RAG Query", "SQL Agent Query"])

with tab1:
    st.title('A  Complete Machine Learning Guide with more than 120 pages PDF')
    st.header("Ask PDF questions")
    query_pdf = st.text_input("Enter your question about the PDF:")

    if st.button("Ask PDF") and query_pdf.strip():
        with st.spinner("Getting answer from PDF RAG..."):
            answer = rag.ask(query_pdf)
            st.success(answer)

with tab2:
    st.header("Ask SQL questions")

    tables = sql_agent.get_tables()
    if isinstance(tables, str):
        st.error(f"Error getting tables: {tables}")
    else:
        selected_table = st.selectbox("Select a table", tables)
        query_sql = st.text_input(f"Ask a question about `{selected_table}`:")

        if st.button("Ask SQL") and query_sql.strip():
            with st.spinner("Getting answer from SQL Agent..."):
                answer = sql_agent.ask(selected_table, query_sql)
                st.success(answer)


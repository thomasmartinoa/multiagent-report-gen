import requests
import streamlit as st

API_URL = "http://localhost:8000/generate"

st.set_page_config(page_title="Research Report Generator", page_icon="")
st.title(" Multi-Agent Research Report Generator")
st.caption("Three specialist agents — Researcher, Writer, Critic — collaborate on a cited report.")

topic = st.text_input("Research topic:", placeholder="e.g. GenAI adoption in Indian fintech")

if st.button("Generate report") and topic:
    with st.spinner("Researcher → Writer → Critic working..."):
        try:
            data = requests.post(API_URL, json={"topic": topic}, timeout=300).json()
        except Exception as e:
            st.error(f"Request failed: {e}")
            st.stop()

    c1, c2 = st.columns(2)
    c1.metric("Revisions", data["revisions"])
    c2.metric("Critic verdict", "Approved" if data["approved"] else "Capped")

    st.markdown(data["report"])

    with st.expander("Critic's review"):
        st.text(data["critique"])

    st.download_button("Download report (.md)", data["report"],
                       file_name="research_report.md")

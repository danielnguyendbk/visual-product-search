"""Streamlit entrypoint for the Visual Product Search skeleton."""

import streamlit as st

st.set_page_config(page_title="Visual Product Search", page_icon="🔎", layout="wide")

st.title("Visual Product Search")
st.caption("Milestone 0 · Project skeleton and REST API contract")

st.info(
    "Search, comparison, and benchmark resources are not ready yet. "
    "Use the pages in the sidebar to inspect the API flow without generated results."
)

st.subheader("Available pages")
st.markdown(
    """
- **Search** uploads a query image and selects Sequential or FAISS.
- **Compare** prepares a side-by-side Sequential and FAISS request.
- **Benchmark** submits benchmark configuration without creating fake measurements.
"""
)

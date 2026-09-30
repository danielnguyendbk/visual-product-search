"""Benchmark page for the Streamlit skeleton."""

import os

import pandas as pd
import requests
import streamlit as st

API_BASE_URL = os.getenv("BACKEND_BASE_URL", "http://127.0.0.1:8000/api/v1")


def show_api_error(response: requests.Response) -> None:
    try:
        error = response.json().get("error", {})
    except requests.JSONDecodeError:
        error = {}

    message = error.get("message", "The backend request failed.")
    code = error.get("code", "UNKNOWN_ERROR")
    if response.status_code == 503:
        st.warning("Resource not ready")
        st.caption(f"{message} Code: {code}")
    else:
        st.error(message)
        st.caption(f"Code: {code}")


st.set_page_config(page_title="Benchmark · Visual Product Search", layout="wide")
st.title("Benchmark")
st.caption("Submit benchmark parameters without generating placeholder measurements.")

dataset_size = st.number_input("Dataset size", min_value=1, value=2000, step=1)
k = st.number_input("Top-K", min_value=1, value=5, step=1)
nprobe = st.number_input("nprobe", min_value=1, value=4, step=1)
num_queries = st.number_input("Number of queries", min_value=1, value=30, step=1)
repeats = st.number_input("Repeats", min_value=1, value=3, step=1)

if st.button("Run Benchmark", type="primary"):
    request_payload = {
        "dataset_size": int(dataset_size),
        "nprobe": int(nprobe),
        "k": int(k),
        "num_queries": int(num_queries),
        "repeats": int(repeats),
    }
    try:
        response = requests.post(
            f"{API_BASE_URL}/benchmark",
            json=request_payload,
            timeout=30,
        )
    except requests.RequestException as exc:
        st.error(f"Could not reach the backend: {exc}")
    else:
        if response.ok:
            st.subheader("Benchmark results")
            st.dataframe(pd.DataFrame([response.json()]), use_container_width=True)
        else:
            show_api_error(response)

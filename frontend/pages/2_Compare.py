"""Comparison page for the Streamlit skeleton."""

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


def metric_value(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.4f}"


st.set_page_config(page_title="Compare · Visual Product Search", layout="wide")
st.title("Compare")
st.caption("The two engines will share one query and the same retrieval configuration.")

uploaded_image = st.file_uploader("Query image", type=["jpg", "jpeg", "png"])
k = st.number_input("Top-K", min_value=1, value=5, step=1)
nprobe = st.number_input("nprobe", min_value=1, value=4, step=1)

if st.button("Run Comparison", type="primary"):
    if uploaded_image is None:
        st.warning("Choose a JPG or PNG image before running the comparison.")
    else:
        files = {
            "image": (
                uploaded_image.name,
                uploaded_image.getvalue(),
                uploaded_image.type,
            )
        }
        try:
            response = requests.post(
                f"{API_BASE_URL}/compare",
                data={"k": int(k), "nprobe": int(nprobe)},
                files=files,
                timeout=30,
            )
        except requests.RequestException as exc:
            st.error(f"Could not reach the backend: {exc}")
        else:
            if response.ok:
                payload = response.json()
                sequential_column, faiss_column = st.columns(2)
                with sequential_column:
                    st.subheader("Sequential")
                    st.dataframe(
                        pd.DataFrame(payload["sequential"]["results"]),
                        use_container_width=True,
                    )
                with faiss_column:
                    st.subheader("FAISS")
                    st.dataframe(
                        pd.DataFrame(payload["faiss"]["results"]),
                        use_container_width=True,
                    )

                overlap, recall, speedup = st.columns(3)
                overlap.metric("Overlap@K", metric_value(payload.get("overlap_at_k")))
                recall.metric("Recall@K", metric_value(payload.get("recall_at_k")))
                speedup.metric("Speedup", metric_value(payload.get("speedup")))
            else:
                show_api_error(response)

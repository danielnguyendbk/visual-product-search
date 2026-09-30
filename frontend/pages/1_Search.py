"""Search page for the Streamlit skeleton."""

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


st.set_page_config(page_title="Search · Visual Product Search", layout="wide")
st.title("Search")
st.caption("Upload a JPG or PNG image and send it to the FastAPI search contract.")

uploaded_image = st.file_uploader("Query image", type=["jpg", "jpeg", "png"])
engine = st.selectbox("Search engine", options=["sequential", "faiss"])
k = st.number_input("Top-K", min_value=1, value=5, step=1)
nprobe = None
if engine == "faiss":
    nprobe = st.number_input("nprobe", min_value=1, value=4, step=1)

if st.button("Search", type="primary"):
    if uploaded_image is None:
        st.warning("Choose a JPG or PNG image before searching.")
    else:
        form_data = {"engine": engine, "k": int(k)}
        if nprobe is not None:
            form_data["nprobe"] = int(nprobe)

        files = {
            "image": (
                uploaded_image.name,
                uploaded_image.getvalue(),
                uploaded_image.type,
            )
        }
        try:
            response = requests.post(
                f"{API_BASE_URL}/search",
                data=form_data,
                files=files,
                timeout=30,
            )
        except requests.RequestException as exc:
            st.error(f"Could not reach the backend: {exc}")
        else:
            if response.ok:
                payload = response.json()
                st.subheader("Results")
                st.write(f"Engine: {payload['engine']} · Top-K: {payload['k']}")
                results = payload.get("results", [])
                if results:
                    st.dataframe(pd.DataFrame(results), use_container_width=True)
                else:
                    st.info("No search results were returned.")
            else:
                show_api_error(response)

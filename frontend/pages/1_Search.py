"""Upload a query and search through FastAPI."""

import streamlit as st

import api_client
from ui import number, preview_query, show_results

st.set_page_config(page_title="Search · Visual Product Search", page_icon="🔎", layout="wide")
st.title("Search")
st.caption("Tra cứu sản phẩm bằng một ảnh query.")

uploaded_image = st.file_uploader("Query image", type=["jpg", "jpeg", "png"])
valid_image = preview_query(uploaded_image)
controls = st.columns(3)
with controls[0]:
    engine = st.selectbox(
        "Engine", ["sequential", "faiss"],
        format_func=lambda value: "Sequential" if value == "sequential" else "FAISS IVF",
    )
with controls[1]:
    k = st.number_input("Top-K", min_value=1, max_value=10, value=5, step=1)
nprobe = 8
if engine == "faiss":
    with controls[2]:
        nprobe = st.number_input("nprobe", min_value=1, max_value=64, value=8, step=1)

st.caption("Chỉ gửi query khi bấm Search. Đổi ảnh hoặc cấu hình sẽ ẩn kết quả cũ.")
if st.button("Search", type="primary", disabled=not valid_image):
    try:
        with st.spinner("Đang encode và tìm kiếm..."):
            payload = api_client.search_image(
                uploaded_image.name, uploaded_image.getvalue(), uploaded_image.type,
                engine=engine, k=int(k), nprobe=int(nprobe),
            )
    except api_client.BackendError as exc:
        st.error(str(exc))
    else:
        st.subheader("Kết quả")
        metrics = st.columns(5 if payload["engine"] == "faiss" else 4)
        metrics[0].metric("Engine", "FAISS IVF" if payload["engine"] == "faiss" else "Sequential")
        metrics[1].metric("Encoding time (ms)", number(payload["encoding_ms"]))
        metrics[2].metric("Search-only latency (ms)", number(payload["search_latency_ms"]))
        metrics[3].metric("K", payload["k"])
        if payload["engine"] == "faiss":
            metrics[4].metric("nprobe", payload["nprobe"])
        st.caption("Search latency không bao gồm thời gian ResNet50 encoding.")
        show_results(payload["results"], columns=3)

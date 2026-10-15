"""Home page for the local Visual Product Search demo."""

import streamlit as st

import api_client

st.set_page_config(page_title="Visual Product Search", page_icon="🔎", layout="wide")

st.title("Visual Product Search")
st.write(
    "Hệ thống tra cứu sản phẩm bằng hình ảnh và so sánh Sequential Search với FAISS IndexIVFFlat."
)

try:
    with st.spinner("Đang kiểm tra backend..."):
        status = api_client.health()
except api_client.BackendError as exc:
    st.error(str(exc))
else:
    backend, catalog, dimension, faiss = st.columns(4)
    backend.metric("Backend status", status["status"])
    catalog.metric("Catalog size", status["catalog_size"])
    dimension.metric("Embedding dimension", status["embedding_dimension"] or "N/A")
    faiss.metric("FAISS ready", "Yes" if status["faiss_ready"] else "No")
    if status["status"] != "ok":
        st.warning("Backend đang chạy nhưng tài nguyên tìm kiếm chưa sẵn sàng.")

sequential_column, faiss_column = st.columns(2)
with sequential_column:
    st.subheader("Sequential Search")
    st.write("Quét toàn bộ vector để lấy Top-K. Đây là exact baseline.")
with faiss_column:
    st.subheader("FAISS IVF")
    st.write("Tìm kiếm trên các phân vùng vector để thực hiện approximate retrieval.")

st.info("Chọn Search, Compare hoặc Benchmark ở sidebar để bắt đầu demo.")

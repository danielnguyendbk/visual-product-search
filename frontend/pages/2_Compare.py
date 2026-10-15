"""Compare both engines with one uploaded query through FastAPI."""

import streamlit as st

import api_client
from ui import number, preview_query, show_results

st.set_page_config(page_title="Compare · Visual Product Search", page_icon="🔎", layout="wide")
st.title("Compare")
st.caption("Hai engine dùng cùng một query; backend chỉ encode ảnh một lần.")

uploaded_image = st.file_uploader("Query image", type=["jpg", "jpeg", "png"])
valid_image = preview_query(uploaded_image)
k_column, nprobe_column = st.columns(2)
with k_column:
    k = st.number_input("Top-K", min_value=1, max_value=10, value=5, step=1)
with nprobe_column:
    nprobe = st.number_input("nprobe", min_value=1, max_value=64, value=8, step=1)

st.caption("Đổi ảnh hoặc cấu hình sẽ ẩn kết quả cũ. Bấm Run Comparison để lấy kết quả mới.")
if st.button("Run Comparison", type="primary", disabled=not valid_image):
    try:
        with st.spinner("Đang encode và so sánh hai engine..."):
            payload = api_client.compare_image(
                uploaded_image.name, uploaded_image.getvalue(), uploaded_image.type,
                k=int(k), nprobe=int(nprobe),
            )
    except api_client.BackendError as exc:
        st.error(str(exc))
    else:
        config, metrics = payload["config"], payload["metrics"]
        latency_columns = st.columns(5)
        latency_columns[0].metric("Sequential latency (ms)", number(payload["sequential"]["latency_ms"]))
        latency_columns[1].metric("FAISS latency (ms)", number(payload["faiss"]["latency_ms"]))
        latency_columns[2].metric("Recall@K", number(metrics["recall_at_k"]))
        latency_columns[3].metric("Overlap", f"{metrics['overlap_count']}/{config['k']}")
        speedup = f"{number(metrics['speedup'])}×" if metrics["speedup"] is not None else "N/A"
        latency_columns[4].metric("Speedup", speedup)
        config_columns = st.columns(4)
        for column, key, label in zip(
            config_columns, ["dataset_size", "k", "nlist", "nprobe"], ["N", "K", "nlist", "nprobe"],
        ):
            column.metric(label, config[key])
        st.caption(
            f"Encoding một lần: {number(payload['query']['encoding_ms'])} ms. "
            "Latency của hai engine chỉ đo search, không bao gồm encoding."
        )
        st.info(
            "Recall@K ở đây là độ trùng của FAISS với exact Sequential Top-K, "
            "không phải độ chính xác phân loại hoặc semantic accuracy."
        )
        for column, engine, title in zip(
            st.columns(2), ["sequential", "faiss"], ["Sequential Search", "FAISS IVF"],
        ):
            with column:
                st.subheader(title)
                show_results(payload[engine]["results"])

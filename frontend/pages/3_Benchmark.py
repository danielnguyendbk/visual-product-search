"""Display saved benchmark aggregates; never run an experiment."""

import pandas as pd
import streamlit as st

import api_client

st.set_page_config(page_title="Benchmark · Visual Product Search", page_icon="🔎", layout="wide")
st.title("Benchmark")
st.caption("Kết quả thí nghiệm đã lưu. Trang này chỉ đọc dữ liệu, không chạy lại benchmark.")

try:
    with st.spinner("Đang tải kết quả benchmark..."):
        payload = api_client.get_benchmark()
except api_client.BackendError as exc:
    st.warning(str(exc))
    st.stop()

metadata = payload["metadata"]
st.subheader("Điều kiện thí nghiệm")
notes = st.columns(4)
for column, field, label in zip(
    notes, ["query_count", "seed", "k", "warmup_queries"], ["Queries", "Seed", "K", "Warm-up queries"],
):
    column.metric(label, metadata.get(field, "N/A"))
if metadata.get("self_exclusion"):
    st.caption(f"Self-match excluded · {metadata['self_exclusion']}")
if metadata.get("sequential_timing") and metadata.get("faiss_timing"):
    st.caption(
        f"Search-only latency · Sequential: {metadata['sequential_timing']}. "
        f"FAISS: {metadata['faiss_timing']}. Không bao gồm ResNet50 encoding."
    )
st.caption(f"Recall@{metadata.get('k', 'K')} so với exact Sequential Top-K, không phải semantic accuracy.")
if metadata.get("build_timing"):
    with st.expander("Cách đo build time"):
        st.write(metadata["build_timing"])
if metadata.get("warnings"):
    for message in metadata["warnings"]:
        st.warning(message)

st.subheader("Experiment A · nprobe")
nprobe_data = pd.DataFrame(payload["nprobe_benchmark"])
nprobe_columns = ["nprobe", "faiss_p50_ms", "faiss_p95_ms", "recall_at_5", "speedup"]
if nprobe_data.empty:
    st.info("Chưa có dữ liệu nprobe.")
elif not set(nprobe_columns).issubset(nprobe_data.columns):
    st.warning("Dữ liệu nprobe thiếu cột cần thiết.")
else:
    nprobe_data = nprobe_data.sort_values("nprobe")
    st.dataframe(nprobe_data[nprobe_columns], hide_index=True, width="stretch")
    latency, recall = st.columns(2)
    with latency:
        st.caption("nprobe → FAISS p50 latency")
        st.line_chart(nprobe_data, x="nprobe", y="faiss_p50_ms", x_label="nprobe", y_label="FAISS p50 (ms)")
    with recall:
        st.caption("nprobe → Recall@5")
        st.line_chart(nprobe_data, x="nprobe", y="recall_at_5", x_label="nprobe", y_label="Recall@5")

st.subheader("Experiment B · Scalability")
scalability_data = pd.DataFrame(payload["scalability_benchmark"])
scalability_columns = [
    "dataset_size", "nlist", "nprobe", "sequential_p50_ms", "faiss_p50_ms",
    "recall_at_5", "speedup", "index_build_ms",
]
if scalability_data.empty:
    st.info("Chưa có dữ liệu scalability.")
elif not set(scalability_columns).issubset(scalability_data.columns):
    st.warning("Dữ liệu scalability thiếu cột cần thiết.")
else:
    scalability_data = scalability_data.sort_values("dataset_size")
    st.dataframe(scalability_data[scalability_columns], hide_index=True, width="stretch")
    st.caption("Dataset size (N) → Search latency (p50)")
    chart_data = scalability_data[["dataset_size", "sequential_p50_ms", "faiss_p50_ms"]].rename(
        columns={"sequential_p50_ms": "Sequential", "faiss_p50_ms": "FAISS IVF"},
    )
    st.line_chart(
        chart_data, x="dataset_size", y=["Sequential", "FAISS IVF"],
        x_label="Dataset size (N)", y_label="Search latency p50 (ms)",
    )

st.caption("Latency và speedup phụ thuộc dữ liệu, cấu hình nprobe và môi trường đo.")

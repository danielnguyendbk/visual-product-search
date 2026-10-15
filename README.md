# Visual Product Search

Visual Product Search compares exact Sequential Search with FAISS `IndexIVFFlat` using the same L2-normalized ResNet50 embeddings. FastAPI reuses the existing encoder and core search services, and serves saved benchmark aggregates.

## Stack

- Backend: FastAPI
- Frontend: Streamlit
- Language: Python
- Search engines: Sequential Search and FAISS `IndexIVFFlat`

## Project structure

```text
visual-product-search/
├── backend/
│   ├── api/              # REST route modules
│   ├── core/             # Settings and shared API exception
│   ├── schemas/          # Pydantic request/response models
│   ├── services/         # Stable service interfaces and temporary catalog
│   └── main.py           # FastAPI app
├── frontend/
│   ├── app.py
│   └── pages/            # Search, Compare, Benchmark
├── data/
│   ├── raw/
│   └── processed/
├── indexes/
├── benchmark_results/
├── tests/
├── requirements.txt
└── README.md
```

Search metadata comes from `data/processed/catalog.csv`, loaded once with the model, embeddings, ID mapping and FAISS index at startup. The legacy `/api/v1/catalog` CRUD service remains a separate temporary in-memory contract; it does not change the search catalog.

## Setup

Create and activate a virtual environment, then install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Dataset and offline preparation

Use **Fashion Product Images (Small)** by Param Aggarwal on Kaggle. Extract it at:

```text
data/raw/fashion-product-images-small/
├── images/
│   └── <id>.jpg
└── styles.csv
```

To create the artifacts from scratch, run these commands from the project root in order:

```powershell
python scripts/prepare_dataset.py --limit 5000
python scripts/build_embeddings.py --batch-size 16
python scripts/build_faiss_index.py
```

Preparation keeps Apparel, Footwear and Accessories with readable images, then shuffles once with seed 42. It writes `data/processed/catalog.csv`. The encoder writes L2-normalized float32 `embeddings.npy` (5000 × 2048), aligned `image_ids.npy` and `encoder_config.json` under `data/processed/`. The index builder writes `indexes/faiss_ivf.index` and `faiss_config.json`: inner product, `IndexIVFFlat`, nlist 64 and default nprobe 8.

These steps replace generated artifacts. Existing artifacts can be used directly for the demo; the API and frontend do not rebuild them.

## Offline benchmark

To reproduce both experiments and replace the saved benchmark outputs:

```powershell
python scripts/run_benchmark.py
```

The experiments use 100 queries, seed 42, K=5, 10 warm-up queries and self-match exclusion. Experiment A varies nprobe on the production IVF64 index. Experiment B uses prefix subsets of 1000, 2000 and 5000 vectors with temporary in-memory indexes. Latencies measure search only. The output directory is `benchmark_results/`: two aggregate CSVs, `benchmark_metadata.json` and `raw_query_results.csv`. The demo reads existing aggregates; running a benchmark is not required to start it.

## Git artifacts

Commit source code, `data/processed/catalog.csv` and the three small benchmark aggregate/metadata files. `.gitkeep` preserves empty directories.

The following generated or local data stay ignored:

- Raw dataset: `data/raw/` (except `.gitkeep`).
- Embeddings, ID mapping and encoder config: `data/processed/` except `catalog.csv` and `.gitkeep`.
- FAISS index and config: `indexes/` except `.gitkeep`.
- Per-query measurements: `benchmark_results/raw_query_results.csv`.
- Python caches, virtual environments and `.env`.

## Run the backend

From the project root:

```powershell
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Swagger UI:

```text
http://127.0.0.1:8000/docs
```

Search, Compare, Benchmark and Health use `/api`; the same handlers are also available under `/api/v1`. Swagger UI documents the multipart fields and response schemas.

Local search requires these prebuilt artifacts:

- `data/processed/catalog.csv`
- `data/processed/embeddings.npy`
- `data/processed/image_ids.npy`
- `indexes/faiss_ivf.index`

ResNet50 uses the existing encoder's torchvision weights cache (or downloads weights on first use). The API does not build embeddings, train indexes or run benchmarks.

## Run the frontend

Keep FastAPI running in terminal 1, then start Streamlit from terminal 2 at the project root:

```powershell
streamlit run frontend/app.py --server.address 127.0.0.1 --server.port 8501
```

Open `http://localhost:8501`. Home displays backend readiness. Search and Compare accept JPG/JPEG/PNG queries and call FastAPI only when their action button is clicked. Changing a control hides the previous results. Product images are read from repository-relative paths; missing images show a short message.

Benchmark displays the saved tables and charts from `GET /api/benchmark`; it never starts an experiment. Search-only latency excludes ResNet50 encoding, and Compare Recall@K measures FAISS agreement with exact Sequential Top-K.

The frontend defaults to `http://127.0.0.1:8000`. To use another backend, set `$env:BACKEND_URL = "http://127.0.0.1:8000"` before starting Streamlit (use the server URL without `/api`).

## API

- `GET /api/health` reports readiness, catalog size and embedding dimension. Startup failures leave health available with `status: "degraded"`.
- `POST /api/search`: multipart `file` (JPG/JPEG/PNG), `engine` (`sequential` or `faiss`, default `sequential`), `k` (default 5), `nprobe` (default 8). Sequential ignores `nprobe` and returns it as `null`.
- `POST /api/compare`: multipart `file`, `k` (default 5), `nprobe` (default 8). Encodes once, calls the existing CompareService, and enriches both result sets with catalog metadata. Recall@K measures ANN agreement with exact Top-K, not semantic accuracy.
- `GET /api/benchmark` reads `nprobe_benchmark.csv`, `scalability_benchmark.csv` and `benchmark_metadata.json` from `benchmark_results/`. Missing files return `503`; the endpoint never reruns experiments.
- `POST /api/v1/index/rebuild` returns `INDEX_NOT_READY` with HTTP `503`.
- Legacy catalog CRUD remains at `/api/v1/catalog`.

Search and Compare retain a matching catalog image in their Top-K. The official benchmark excludes the query image itself, so its Top-K sets and Recall@5 can differ from an interactive catalog-image query.

Search returns `encoding_ms` separately from `search_latency_ms`. Search latency is taken directly from the core engine timer and excludes encoding, artifact loading and metadata lookup. Compare preserves the core's per-engine latency and metrics. Requests share a lock to protect mutable FAISS settings and core latency fields; these timings are request observations, while the saved experiments remain the official benchmark.

Invalid images, engines, K or FAISS nprobe return `400`. Search/Compare return `503` when shared resources are unavailable. Uploaded images are decoded to RGB in memory and are not saved permanently.

Example from PowerShell (replace the image path with an existing local image):

```powershell
curl.exe -X POST http://127.0.0.1:8000/api/search -F "file=@data/raw/fashion-product-images-small/images/31893.jpg" -F "engine=faiss" -F "k=5" -F "nprobe=8"
curl.exe -X POST http://127.0.0.1:8000/api/compare -F "file=@data/raw/fashion-product-images-small/images/31893.jpg" -F "k=5" -F "nprobe=8"
curl.exe http://127.0.0.1:8000/api/benchmark
```

All API errors use this envelope:

```json
{
  "error": {
    "code": "SEARCH_ENGINE_NOT_READY",
    "message": "Search resources are unavailable. Check the model, catalog and artifacts."
  }
}
```

## Tests

```powershell
python -m pytest -q tests/test_api_contract.py tests/test_health.py
python -m pytest -q
```

For the current Windows environment, use the writable temp directory and disable pytest's cache provider:

```powershell
python -m pytest -q --basetemp="C:\Users\HP\pytest-vps-temp\run" -p no:cacheprovider
```

API tests use lightweight encoder/search doubles with the real integration layer and CompareService, so they need no model download or local dataset. They cover metadata, shared-resource reuse, one-time query encoding, core timings, input validation, readiness, read-only benchmark loading and legacy catalog CRUD. `pytest.ini` collects `tests/`; scripts under `scripts/test_*.py` remain manual checks.

Frontend unit tests cover the HTTP upload contract, backend failures and safe local image resolution. They do not require a running backend.

## Planned ownership

- Member 1: Data, Encoder, Catalog, Search UI
- Member 2: Sequential Search, FAISS, Search/Compare engine
- Member 3: Benchmark, Integration, Tests

The existing core services remain the source of encoding, search and comparison logic.

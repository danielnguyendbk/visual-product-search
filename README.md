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

## Run the backend

From the project root:

```powershell
python -m uvicorn backend.main:app --reload
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

Keep the backend running, then start Streamlit from another terminal:

```powershell
streamlit run frontend/app.py
```

The frontend is the existing mock flow and has not yet been integrated with these API responses. It uses `http://127.0.0.1:8000/api/v1` by default; set `BACKEND_BASE_URL` to override it.

## API

- `GET /api/health` reports readiness, catalog size and embedding dimension. Startup failures leave health available with `status: "degraded"`.
- `POST /api/search`: multipart `file` (JPG/JPEG/PNG), `engine` (`sequential` or `faiss`, default `sequential`), `k` (default 5), `nprobe` (default 8). Sequential ignores `nprobe` and returns it as `null`.
- `POST /api/compare`: multipart `file`, `k` (default 5), `nprobe` (default 8). Encodes once, calls the existing CompareService, and enriches both result sets with catalog metadata. Recall@K measures ANN agreement with exact Top-K, not semantic accuracy.
- `GET /api/benchmark` reads `nprobe_benchmark.csv`, `scalability_benchmark.csv` and `benchmark_metadata.json` from `benchmark_results/`. Missing files return `503`; the endpoint never reruns experiments.
- `POST /api/v1/index/rebuild` returns `INDEX_NOT_READY` with HTTP `503`.
- Legacy catalog CRUD remains at `/api/v1/catalog`.

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

API tests use lightweight encoder/search doubles with the real integration layer and CompareService, so they need no model download or local dataset. They cover metadata, shared-resource reuse, one-time query encoding, core timings, input validation, readiness, read-only benchmark loading and legacy catalog CRUD. `pytest.ini` collects `tests/`; scripts under `scripts/test_*.py` remain manual checks.

## Planned ownership

- Member 1: Data, Encoder, Catalog, Search UI
- Member 2: Sequential Search, FAISS, Search/Compare engine
- Member 3: Benchmark, Integration, Tests

The existing core services remain the source of encoding, search and comparison logic.

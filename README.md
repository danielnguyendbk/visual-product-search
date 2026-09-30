# Visual Product Search

Milestone 0 provides the project skeleton, REST API contract, temporary in-memory catalog, and Streamlit mock flow for a future visual product search system.

No encoder, embeddings, dataset, Sequential Search implementation, FAISS index, or benchmark measurement exists in this milestone. Search, comparison, benchmark, and index rebuild requests therefore return an explicit HTTP `503` error instead of generated data.

## Stack

- Backend: FastAPI
- Frontend: Streamlit
- Language: Python
- Future search engines: Sequential Search and FAISS `IndexIVFFlat`

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

The catalog service stores records only in process memory. Data is lost whenever the backend restarts. It is a temporary implementation for API contract tests, not a production database.

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

All application routes use the `/api/v1` base path.

## Run the frontend

Keep the backend running, then start Streamlit from another terminal:

```powershell
streamlit run frontend/app.py
```

The frontend uses `http://127.0.0.1:8000/api/v1` by default. Set `BACKEND_BASE_URL` to override it.

## API readiness behavior

- `GET /api/v1/health` reports model, embeddings, and index readiness as `false`.
- `POST /api/v1/search` returns `SEARCH_ENGINE_NOT_READY` with HTTP `503` after validating the request.
- `POST /api/v1/compare` returns HTTP `503` while shared search resources are unavailable.
- `POST /api/v1/benchmark` returns `BENCHMARK_NOT_READY` with HTTP `503`.
- `POST /api/v1/index/rebuild` returns `INDEX_NOT_READY` with HTTP `503`.
- Catalog CRUD is available through the temporary in-memory service.

All API errors use this envelope:

```json
{
  "error": {
    "code": "SEARCH_ENGINE_NOT_READY",
    "message": "Search engine is not ready."
  }
}
```

## Tests

```powershell
python -m pytest
```

The contract suite covers health, OpenAPI discovery, image and parameter validation, unavailable search and benchmark resources, and catalog CRUD.

## Planned ownership

- Member 1: Data, Encoder, Catalog, Search UI
- Member 2: Sequential Search, FAISS, Search/Compare engine
- Member 3: Benchmark, Integration, Tests

The service signatures in `backend/services/` are reserved for these later milestones and should not be changed without a team decision.

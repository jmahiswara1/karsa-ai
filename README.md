# Karsa AI Service

Intelligence layer for Karsa Productivity Assistant.

## Setup

1. Install dependencies using `uv`:
   ```bash
   uv venv
   uv pip install -e .
   ```

2. Setup environment variables (copy `.env.example` to `.env`).

3. Run server:
   ```bash
   uv run uvicorn app.main:app --reload
   ```

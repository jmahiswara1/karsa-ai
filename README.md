# Karsa AI Service

Intelligence layer for Karsa Productivity Assistant.

## Setup

1. Buat virtual environment dan install dependencies:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -e .[dev]
   ```

2. Setup environment variables (pastikan file `.env` sudah ada dan memiliki key yang benar, bisa merujuk ke `.env.example`).

3. Run server:
   ```powershell
   .\.venv\Scripts\Activate.ps1
   uvicorn app.main:app --reload --port 8000
   ```

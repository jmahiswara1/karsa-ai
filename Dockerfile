FROM python:3.11-slim

WORKDIR /app

RUN pip install uv

COPY . .
RUN uv pip install --system .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

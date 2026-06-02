from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router as ai_router

app = FastAPI(title="Karsa AI Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ai_router, prefix="/api", tags=["AI"])

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "ai-karsa"}

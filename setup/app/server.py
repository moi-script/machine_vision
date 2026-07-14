"""FastAPI application entrypoint for AeroSense."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import db

app = FastAPI(title="AeroSense Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from app.routers import players

app.include_router(players.router)


@app.get("/api/health")
def health():
    return {"mongo": db.ping(), "engine": "idle", "camera": "unknown"}

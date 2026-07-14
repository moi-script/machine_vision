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

from app.routers import players, sessions, settings as settings_router, control

app.include_router(players.router)
app.include_router(sessions.router)
app.include_router(settings_router.router)
app.include_router(control.router)
app.include_router(control.media_router)


@app.get("/api/health")
def health():
    from app.engine import get_engine
    st = get_engine().status()
    return {"mongo": db.ping(), "engine": st["state"], "camera": st["camera"]}

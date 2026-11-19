from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from threading import Lock
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, StringConstraints
from typing_extensions import Annotated

from . import store
from .model import ROOT, cues, examples, metrics, probabilities

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=3, max_length=2500)]


class Message(BaseModel):
    text: Text


class Review(Message):
    label: Literal["scam", "genuine", "unsure"]


class Replay(BaseModel):
    cohort: Literal["audit", "review"] = "audit"
    candidate: int | None = None


@asynccontextmanager
async def lifespan(application):
    store.initialize()
    with ThreadPoolExecutor(max_workers=1) as executor:
        application.state.training_executor = executor
        application.state.training_job = None
        application.state.training_lock = Lock()
        yield


app = FastAPI(title="ScamShift", lifespan=lifespan)


@app.get("/api/status")
def status():
    return store.status()


@app.post("/api/scan")
def scan(payload: Message):
    version, model, meta = store.load()
    score = float(probabilities(model, [payload.text])[0])
    flagged = score >= meta["threshold"]
    with store.db() as conn:
        conn.execute("INSERT INTO scans(model_id,flagged) VALUES(?,?)", (version, flagged))
    return {"model": version, "score": score, "flagged": flagged, "cues": cues(payload.text)}


@app.post("/api/replay")
def replay(payload: Replay):
    rows = examples(payload.cohort)
    texts = [row["text"] for row in rows]
    labels = [row["label"] for row in rows]
    active, _, _ = store.load()
    versions = [active]
    if payload.candidate is not None and payload.candidate != active:
        versions.append(payload.candidate)
    comparison = []
    for version in versions:
        try:
            _, model, meta = store.load(version)
        except LookupError as exc:
            raise HTTPException(404, str(exc)) from exc
        scores = probabilities(model, texts)
        comparison.append({
            "version": version,
            "metrics": metrics(labels, scores, meta["threshold"]),
            "scores": scores.tolist(),
            "flags": (scores >= meta["threshold"]).tolist(),
        })
    with store.db() as conn:
        reviewed = {row["id"]: row["label"] for row in conn.execute("SELECT id,label FROM reviews")}
    for row in rows:
        row["reviewed"] = row["id"] in reviewed
        row["review_label"] = reviewed.get(row["id"])
    return {"rows": rows, "comparison": comparison}


@app.post("/api/reviews")
def review(payload: Review):
    try:
        store.review(payload.text, {"scam": 1, "genuine": 0, "unsure": None}[payload.label])
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    return {"saved": True}


@app.post("/api/train", status_code=202)
def train():
    with app.state.training_lock:
        job = app.state.training_job
        if job is not None and not job.done():
            raise HTTPException(409, "Training is already in progress.")
        if not store.status()["reviewed"]:
            raise HTTPException(409, "Review at least one message before training.")
        app.state.training_job = app.state.training_executor.submit(store.register)
    return {"state": "training"}


@app.get("/api/train")
def training_status():
    job = app.state.training_job
    if job is None:
        return {"state": "idle"}
    if not job.done():
        return {"state": "training"}
    try:
        return {"state": "finished", "version": job.result()}
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(500, "Training failed. Check the server logs.") from exc


@app.post("/api/models/{version}/{action}")
def activate(version: int, action: Literal["promote", "rollback"]):
    try:
        store.activate(version, rollback=action == "rollback")
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    return {"active": version}


app.mount("/", StaticFiles(directory=ROOT / "web", html=True), name="dashboard")

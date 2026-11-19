from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from time import perf_counter
from typing import Literal
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, StringConstraints
from starlette.middleware.trustedhost import TrustedHostMiddleware
from typing_extensions import Annotated

from . import registry, store
from .datasets import REVIEW_LABELS
from .drift import priority_band
from .evaluation import sliced_metrics
from .model import ROOT, cues, examples, metrics, probabilities

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=3, max_length=2500)]


class Message(BaseModel):
    text: Text


class Review(Message):
    label: Literal["scam", "genuine", "promotional_spam", "unsure"]


class Replay(BaseModel):
    cohort: Literal["audit", "review", "robustness"] = "audit"
    candidate: int | None = None


class DriftOptions(BaseModel):
    days: Annotated[int, Field(ge=1, le=30)] = 7
    minimum: Annotated[int, Field(ge=10, le=10000)] = 30


class LimitAPIInput:
    """Bound API write bodies before parsing, including chunked requests."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or not scope["path"].startswith("/api/") or scope["method"] != "POST":
            return await self.app(scope, receive, send)
        body = bytearray()
        while True:
            event = await receive()
            if event["type"] == "http.disconnect":
                return
            chunk = event.get("body", b"")
            if len(body) + len(chunk) > 16_384:
                return await JSONResponse({"detail": "Request body exceeds 16 KiB."}, status_code=413)(
                    scope, receive, send)
            body.extend(chunk)
            if not event.get("more_body", False):
                break
        first = True

        async def bounded_receive():
            nonlocal first
            if first:
                first = False
                return {"type": "http.request", "body": bytes(body), "more_body": False}
            return await receive()

        return await self.app(scope, bounded_receive, send)


@asynccontextmanager
async def lifespan(application):
    store.initialize()
    with ThreadPoolExecutor(max_workers=1) as executor:
        application.state.training_executor = executor
        yield


app = FastAPI(title="ScamShift", lifespan=lifespan)
app.add_middleware(LimitAPIInput)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost"])


@app.middleware("http")
async def request_context(request: Request, call_next):
    request_id = uuid4().hex
    if request.method in {"POST", "PUT", "PATCH", "DELETE"}:
        origin = request.headers.get("origin")
        expected = f"{request.url.scheme}://{request.url.netloc}"
        if (origin is not None and origin != expected) or request.headers.get("sec-fetch-site") == "cross-site":
            return JSONResponse({"detail": "Cross-origin writes are blocked."}, status_code=403,
                                headers={"X-Request-ID": request_id})
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


@app.exception_handler(RequestValidationError)
async def invalid_request(request: Request, exc: RequestValidationError):
    return JSONResponse(status_code=422, content={
        "detail": "Request validation failed.",
        "errors": [{"field": ".".join(map(str, item["loc"])), "message": item["msg"]} for item in exc.errors()],
    })


@app.get("/api/health")
@app.get("/api/readiness")
def health():
    try:
        active, _, _ = store.load()
    except (LookupError, store.ArtifactError) as exc:
        raise HTTPException(503, "Active model unavailable. Restore a verified model in the registry.") from exc
    return {"state": "ready", "active": active}


@app.get("/api/status")
def status():
    return store.status()


@app.post("/api/registry/reconcile")
def reconcile_registry():
    try:
        return {"active": store.reconcile_registry()}
    except registry.RegistryError as exc:
        raise HTTPException(503, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc


@app.post("/api/scan")
def scan(payload: Message):
    started = perf_counter()
    try:
        version, model, meta = store.load()
    except (LookupError, store.ArtifactError) as exc:
        raise HTTPException(503, "Active model unavailable. Restore a verified model in the registry.") from exc
    score = float(probabilities(model, [payload.text])[0])
    flagged = score >= meta["threshold"]
    band = priority_band(score, meta["threshold"])
    matched = cues(payload.text)
    store.record_scan(version, payload.text, score, flagged, band, (perf_counter() - started) * 1000)
    return {"model": version, "score": score, "threshold": meta["threshold"], "flagged": flagged,
            "priority": band, "cues": matched}


@app.post("/api/replay")
def replay(payload: Replay):
    rows = examples(payload.cohort)
    texts = [row["text"] for row in rows]
    labels = [row["label"] for row in rows]
    active = store.current()
    versions = [active]
    if payload.candidate is not None and payload.candidate != active:
        versions.append(payload.candidate)
    comparison = []
    for version in versions:
        try:
            _, model, meta = store.load(version)
        except LookupError as exc:
            raise HTTPException(404, str(exc)) from exc
        except store.ArtifactError as exc:
            raise HTTPException(503, "A model artifact is unavailable.") from exc
        scores = probabilities(model, texts)
        comparison.append({
            "version": version,
            "metrics": metrics(labels, scores, meta["threshold"]),
            "by_campaign": sliced_metrics(rows, scores, meta["threshold"], "campaign"),
            "by_script": sliced_metrics(rows, scores, meta["threshold"], "script"),
            "scores": scores.tolist(),
            "flags": (scores >= meta["threshold"]).tolist(),
        })
    if payload.cohort == "review":
        reviewed = store.review_status([row["id"] for row in rows])
        for row in rows:
            saved = reviewed.get(row["id"])
            row["reviewed"] = saved is not None
            row["review_label"] = saved["label"] if saved else None
            row["review_updated"] = saved["updated"] if saved else None
            row["review_history"] = saved["history"] if saved else []
    return {"rows": rows, "comparison": comparison}


@app.post("/api/reviews")
def review(payload: Review):
    try:
        updated = store.review(payload.text, REVIEW_LABELS[payload.label])
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    return {"saved": True, "updated": updated}


@app.get("/api/drift")
def drift_history():
    return {"reports": store.drift_history()}


@app.post("/api/drift")
def drift_report(options: DriftOptions):
    return store.save_drift_report(options.days, options.minimum)


@app.post("/api/train", status_code=202)
def train():
    try:
        job_id, reviewed = store.start_training()
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    try:
        app.state.training_executor.submit(store.run_training, job_id, reviewed)
    except RuntimeError as exc:
        store.fail_training(job_id)
        raise HTTPException(503, "Could not start training. Retry shortly.") from exc
    return {"state": "training", "id": job_id}


@app.get("/api/train")
def training_status():
    return store.training_status() or {"state": "idle"}


@app.get("/api/train/{job_id}")
def training_job(job_id: int):
    job = store.training_status(job_id)
    if job is None:
        raise HTTPException(404, "Unknown training job.")
    return job


@app.post("/api/models/{version}/{action}")
def activate(version: int, action: Literal["promote", "rollback"]):
    try:
        store.activate(version, rollback=action == "rollback")
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc
    except registry.RegistryError as exc:
        raise HTTPException(503, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    return {"active": version}


app.mount("/", StaticFiles(directory=ROOT / "web", html=True), name="dashboard")

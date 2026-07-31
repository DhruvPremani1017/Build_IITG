"""Inference API for the AI-Powered Glaucoma Risk Prediction hackathon.

Implements the contract in Section 7 of the problem statement:
  POST /predict   { "inputFile": "<https url to a csv>" }
  -> downloads the input file, scores every row (order preserved), writes
     an output CSV, and returns a JSON envelope pointing at it.
  GET  /download/{filename}  -> serves a previously generated output CSV.

The documented contract is CSV-only, but the hackathon's generic endpoint
tester advertises probing with CSV/TXT/JSON across problems, and the
rubric explicitly scores graceful handling of unexpected input. So input
parsing tries CSV -> delimiter-sniffed text (handles .txt files that are
tab/semicolon-delimited) -> JSON, and only fails cleanly (400) if none of
those work.
"""
import io
import json
import logging
import os
import uuid
from datetime import datetime, timezone

import joblib
import pandas as pd
import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

from feature_engineering import (  # noqa: F401  (needed to unpickle model.pkl)
    BlendedGlaucomaModel,
    ClinicalPriorScorer,
    GlaucomaFeatureEngineer,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("glaucoma-api")

MODEL_PATH = os.environ.get("MODEL_PATH", "model.pkl")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "outputs")
DOWNLOAD_TIMEOUT_SECONDS = 30
MAX_DOWNLOAD_BYTES = 200 * 1024 * 1024  # 200 MB guard against runaway downloads

os.makedirs(OUTPUT_DIR, exist_ok=True)

app = FastAPI(title="Glaucoma Risk Prediction API")

model = joblib.load(MODEL_PATH)


class PredictRequest(BaseModel):
    inputFile: str = Field(..., description="Publicly accessible HTTP(S) URL of the input file")


def _error_envelope(message: str):
    return {"data": None, "message": message, "status": "error"}


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.warning("Request validation failed: %s", exc)
    return JSONResponse(status_code=422, content=_error_envelope(f"Invalid request: {exc.errors()}"))


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    return JSONResponse(status_code=exc.status_code, content=_error_envelope(str(exc.detail)))


def _records_from_json(payload) -> list:
    """Normalize a few common JSON shapes into a list-of-row-dicts."""
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("data", "records", "rows", "items"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
        return [payload]  # treat as a single row
    raise ValueError("Unsupported JSON structure")


def _decode_text(content: bytes) -> str:
    """utf-8-sig strips a BOM if present and is otherwise identical to utf-8
    (common with Excel-exported CSVs). Falls back to latin-1, which never
    raises a decode error, as a last resort for unexpected encodings."""
    try:
        return content.decode("utf-8-sig")
    except UnicodeDecodeError:
        return content.decode("latin-1")


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _parse_input_bytes(content: bytes) -> pd.DataFrame:
    text = _decode_text(content)
    stripped = text.lstrip()
    looks_like_json = stripped[:1] in ("{", "[")

    if looks_like_json:
        try:
            records = _records_from_json(json.loads(text))
            df = pd.DataFrame(records)
            if df.shape[1] > 0:
                return _normalize_columns(df)
        except Exception:  # noqa: BLE001 - fall through to delimited-text attempts
            pass

    # Standard comma CSV (the documented contract).
    try:
        df = pd.read_csv(io.StringIO(text))
        if df.shape[1] > 1 or looks_like_json is False:
            return _normalize_columns(df)
    except Exception:  # noqa: BLE001
        df = None

    # .txt-style files with a different delimiter (tab/semicolon/pipe) — auto-sniff.
    try:
        df_sniffed = pd.read_csv(io.StringIO(text), sep=None, engine="python")
        if df_sniffed.shape[1] >= 1:
            return _normalize_columns(df_sniffed)
    except Exception:  # noqa: BLE001
        pass

    if df is not None:
        return _normalize_columns(df)

    # Last resort: JSON, even if it didn't look like JSON up front.
    try:
        records = _records_from_json(json.loads(text))
        return _normalize_columns(pd.DataFrame(records))
    except Exception:  # noqa: BLE001
        pass

    raise ValueError("Could not parse inputFile as CSV, delimited text, or JSON")


def _download_input(url: str) -> pd.DataFrame:
    if not url.lower().startswith(("http://", "https://")):
        raise ValueError("inputFile must be an HTTP(S) URL")
    try:
        resp = requests.get(url, timeout=DOWNLOAD_TIMEOUT_SECONDS, stream=True)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise ValueError(f"Failed to download inputFile: {exc}") from exc

    content = bytearray()
    for chunk in resp.iter_content(chunk_size=1024 * 256):
        content.extend(chunk)
        if len(content) > MAX_DOWNLOAD_BYTES:
            raise ValueError("inputFile exceeds maximum allowed size")

    if len(content) == 0:
        raise ValueError("inputFile is empty")

    try:
        df = _parse_input_bytes(bytes(content))
    except Exception as exc:  # noqa: BLE001 - want a clean 400, not a 500 traceback
        raise ValueError(f"Could not parse inputFile: {exc}") from exc

    if df.shape[1] == 0:
        raise ValueError("inputFile has no columns")
    return df


def _run_inference(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    if len(df) == 0:
        for col in ("Diagnosis", "Glaucoma_Probability", "Risk_Band", "Triage_Recommendation"):
            result[col] = pd.Series(dtype="object")
        return result

    features = df.drop(columns=["Diagnosis"], errors="ignore")
    bundle = model.predict_bundle(features)
    bundle.index = df.index

    # Required by spec Section 7.2/7.3: boolean decision in text form.
    result["Diagnosis"] = bundle["label"].map({1: "true", 0: "false"})
    # Additional columns for goal 2.1 ("confidence score and risk band, not
    # just a binary label") -- appended after all required fields so they
    # never interfere with contract-compliance checks on the required schema.
    result["Glaucoma_Probability"] = bundle["probability"].round(6)
    result["Risk_Band"] = bundle["risk_band"]
    result["Triage_Recommendation"] = bundle["triage"]
    return result


@app.post("/predict")
def predict(payload: PredictRequest, request: Request):
    try:
        input_df = _download_input(payload.inputFile)
        row_count = len(input_df)
        output_df = _run_inference(input_df)

        if len(output_df) != row_count:
            raise ValueError("Internal error: output row count does not match input row count")

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"glaucoma_predictions_{timestamp}_{uuid.uuid4().hex[:8]}.csv"
        filepath = os.path.join(OUTPUT_DIR, filename)
        output_df.to_csv(filepath, index=False)

        base_url = os.environ.get("PUBLIC_BASE_URL", "").rstrip("/") or str(request.base_url).rstrip("/")
        output_file_url = f"{base_url}/download/{filename}"

        return {
            "data": {
                "outputFile": output_file_url,
                "timestamp": timestamp,
            },
            "message": "Predictions generated successfully",
            "status": "success",
        }
    except ValueError as exc:
        logger.warning("predict() rejected request: %s", exc)
        return JSONResponse(status_code=400, content=_error_envelope(str(exc)))
    except Exception as exc:  # noqa: BLE001
        logger.exception("predict() failed unexpectedly")
        return JSONResponse(status_code=500, content=_error_envelope(f"Internal error: {exc}"))


@app.get("/download/{filename}")
def download(filename: str):
    safe_name = os.path.basename(filename)
    filepath = os.path.join(OUTPUT_DIR, safe_name)
    if not os.path.isfile(filepath):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(filepath, media_type="text/csv", filename=safe_name)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

from __future__ import annotations

import io
import os
import time
import threading
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from bson import ObjectId
from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from pymongo import ASCENDING, MongoClient
import gridfs

load_dotenv()

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

MONGODB_URI = os.getenv("MONGODB_URI")
MONGODB_DB = os.getenv("MONGODB_DB", "data_cleaner")
UPLOAD_TTL_MINUTES = int(os.getenv("UPLOAD_TTL_MINUTES", "10"))
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "25"))
CORS_ORIGIN = os.getenv("CORS_ORIGIN", "*")
PORT = int(os.getenv("PORT", "5000"))

MAX_CONTENT_LENGTH = MAX_UPLOAD_MB * 1024 * 1024

if not MONGODB_URI:
    raise RuntimeError("MONGODB_URI is not set")

# -----------------------------------------------------------------------------
# App + DB
# -----------------------------------------------------------------------------

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
CORS(app, resources={r"/*": {"origins": CORS_ORIGIN}})

client = MongoClient(MONGODB_URI)
db = client[MONGODB_DB]
fs = gridfs.GridFS(db)

meta = db["file_metadata"]
cleaned_meta = db["cleaned_file_metadata"]

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def utc_now() -> datetime:
    return datetime.now(timezone.utc)

def expiration_time() -> datetime:
    return utc_now() + timedelta(minutes=UPLOAD_TTL_MINUTES)

def oid(value: str) -> ObjectId:
    return ObjectId(value)

def json_error(message: str, code: int = 400):
    return jsonify({"error": message}), code

def safe_ext(filename: str) -> str:
    return os.path.splitext(filename)[1].lower().strip()

def safe_preview(df: pd.DataFrame, limit: int = 50) -> List[Dict[str, Any]]:
    preview = df.head(limit).copy().astype(object)
    preview = preview.where(pd.notnull(preview), None)

    records: List[Dict[str, Any]] = []
    for row in preview.to_dict(orient="records"):
        clean_row: Dict[str, Any] = {}
        for key, value in row.items():
            if isinstance(value, np.generic):
                value = value.item()
            if isinstance(value, np.ndarray):
                value = value.tolist()
            if value is None:
                clean_row[key] = None
            else:
                try:
                    clean_row[key] = None if pd.isna(value) else value
                except Exception:
                    clean_row[key] = value
        records.append(clean_row)
    return records

def read_dataframe_from_bytes(data: bytes, filename: str) -> pd.DataFrame:
    ext = safe_ext(filename)
    bio = io.BytesIO(data)

    if ext == ".csv":
        return pd.read_csv(bio)
    if ext == ".xlsx":
        return pd.read_excel(bio, engine="openpyxl")
    if ext == ".xls":
        return pd.read_excel(bio, engine="xlrd")

    raise ValueError("Unsupported file format. Use CSV, XLSX, or XLS.")

def dataframe_to_bytes(df: pd.DataFrame, original_filename: str) -> Tuple[bytes, str, str]:
    ext = safe_ext(original_filename)

    if ext == ".csv":
        out = io.StringIO()
        df.to_csv(out, index=False)
        return out.getvalue().encode("utf-8"), "text/csv", ".csv"

    if ext in {".xlsx", ".xls"}:
        out = io.BytesIO()
        df.to_excel(out, index=False, engine="openpyxl")
        return (
            out.getvalue(),
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            ".xlsx",
        )

    raise ValueError("Unsupported export format")

def validate_object_id(file_id: str) -> ObjectId:
    try:
        return oid(file_id)
    except Exception as exc:
        raise ValueError("Invalid file id") from exc

def find_file_doc(file_id: str):
    _id = validate_object_id(file_id)
    for collection in (meta, cleaned_meta):
        doc = collection.find_one({"file_id": _id})
        if doc:
            return collection, doc
    raise FileNotFoundError("File not found or expired")

def delete_gridfs_file(file_id: ObjectId) -> None:
    try:
        fs.delete(file_id)
    except Exception:
        pass

def cleanup_expired_files() -> None:
    """
    Deletes expired GridFS files and their metadata.
    Expiry is driven by fs.files.metadata.expires_at.
    """
    now = utc_now()

    expired_files = list(db["fs.files"].find({"metadata.expires_at": {"$lte": now}}, {"_id": 1}))
    for item in expired_files:
        file_id = item.get("_id")
        if not isinstance(file_id, ObjectId):
            continue

        delete_gridfs_file(file_id)
        try:
            meta.delete_many({"file_id": file_id})
            cleaned_meta.delete_many({"file_id": file_id})
        except Exception:
            pass

def reaper_loop() -> None:
    while True:
        try:
            cleanup_expired_files()
        except Exception:
            pass
        time.sleep(5)

def start_reaper_once() -> None:
    threading.Thread(target=reaper_loop, daemon=True).start()

def clean_data_frame(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    result = df.copy()

    if config.get("remove_nulls"):
        result = result.dropna()

    if config.get("drop_duplicates"):
        result = result.drop_duplicates()

    normalize_cols = config.get("normalize_text", [])
    if isinstance(normalize_cols, str):
        normalize_cols = [normalize_cols]

    for col in normalize_cols:
        if col in result.columns:
            result[col] = result[col].astype(str).str.lower().str.strip()

    return result

def store_uploaded_file(file_storage):
    if "file" not in request.files:
        return json_error("No file part in request")

    if not file_storage or not file_storage.filename:
        return json_error("No file selected")

    filename = file_storage.filename
    ext = safe_ext(filename)
    if ext not in {".csv", ".xlsx", ".xls"}:
        return json_error("Unsupported file type. Allowed: CSV, XLSX, XLS")

    raw = file_storage.read()
    if not raw:
        return json_error("Empty file")

    df = read_dataframe_from_bytes(raw, filename)

    expires_at = expiration_time()
    created_at = utc_now()

    gridfs_id = fs.put(
        raw,
        filename=filename,
        contentType=file_storage.mimetype or "application/octet-stream",
        metadata={
            "kind": "original",
            "created_at": created_at,
            "expires_at": expires_at,
            "rows": int(df.shape[0]),
            "cols": int(df.shape[1]),
        },
    )

    meta.insert_one(
        {
            "_id": ObjectId(),
            "file_id": gridfs_id,
            "filename": filename,
            "created_at": created_at,
            "expires_at": expires_at,
            "rows": int(df.shape[0]),
            "cols": int(df.shape[1]),
            "kind": "original",
        }
    )

    return jsonify(
        {
            "status": "success",
            "file_id": str(gridfs_id),
            "filename": filename,
            "expires_at": expires_at.isoformat(),
            "columns": list(df.columns),
            "preview": safe_preview(df),
        }
    )

# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------

@app.get("/health")
def health():
    return jsonify({"status": "ok", "time": utc_now().isoformat()})

@app.post("/upload")
@app.post("/load-file")
def upload_or_load_file():
    try:
        if "file" not in request.files:
            return json_error("No file part in request")

        return store_uploaded_file(request.files["file"])
    except pd.errors.EmptyDataError:
        return json_error("The uploaded file contains no data")
    except Exception as exc:
        return json_error(str(exc), 500)

@app.get("/file/<file_id>")
def get_file_info(file_id: str):
    try:
        _, doc = find_file_doc(file_id)
        return jsonify(
            {
                "file_id": str(doc["file_id"]),
                "filename": doc.get("filename"),
                "created_at": doc.get("created_at").isoformat() if doc.get("created_at") else None,
                "expires_at": doc.get("expires_at").isoformat() if doc.get("expires_at") else None,
                "rows": doc.get("rows"),
                "cols": doc.get("cols"),
                "kind": doc.get("kind"),
            }
        )
    except FileNotFoundError as exc:
        return json_error(str(exc), 404)
    except Exception as exc:
        return json_error(str(exc), 500)

@app.post("/clean/<file_id>")
@app.post("/clean-data")
def clean_file(file_id: str | None = None):
    try:
        payload = request.get_json(silent=True) or {}

        if file_id is None:
            file_id = payload.get("file_id")

        if not file_id:
            return json_error("Missing file_id")

        if "config" in payload and isinstance(payload["config"], dict):
            config = payload["config"]
        else:
            config = payload if isinstance(payload, dict) else {}

        _, doc = find_file_doc(file_id)

        blob = fs.get(doc["file_id"]).read()
        source_df = read_dataframe_from_bytes(blob, doc["filename"])
        cleaned_df = clean_data_frame(source_df, config)

        cleaned_bytes, mime, out_ext = dataframe_to_bytes(cleaned_df, doc["filename"])
        expires_at = expiration_time()
        created_at = utc_now()
        cleaned_name = os.path.splitext(doc["filename"])[0] + f"_cleaned{out_ext}"

        cleaned_gridfs_id = fs.put(
            cleaned_bytes,
            filename=cleaned_name,
            contentType=mime,
            metadata={
                "kind": "cleaned",
                "source_file_id": doc["file_id"],
                "created_at": created_at,
                "expires_at": expires_at,
                "rows": int(cleaned_df.shape[0]),
                "cols": int(cleaned_df.shape[1]),
            },
        )

        cleaned_meta.insert_one(
            {
                "_id": ObjectId(),
                "file_id": cleaned_gridfs_id,
                "source_file_id": doc["file_id"],
                "filename": cleaned_name,
                "created_at": created_at,
                "expires_at": expires_at,
                "rows": int(cleaned_df.shape[0]),
                "cols": int(cleaned_df.shape[1]),
                "kind": "cleaned",
            }
        )

        return jsonify(
            {
                "status": "success",
                "file_id": str(cleaned_gridfs_id),
                "filename": cleaned_name,
                "expires_at": expires_at.isoformat(),
                "columns": list(cleaned_df.columns),
                "preview": safe_preview(cleaned_df),
            }
        )
    except FileNotFoundError as exc:
        return json_error(str(exc), 404)
    except Exception as exc:
        return json_error(str(exc), 500)

@app.get("/download/<file_id>")
@app.get("/export-file/<file_id>")
def export_file(file_id: str):
    try:
        _, doc = find_file_doc(file_id)
        blob = fs.get(doc["file_id"])
        filename = doc.get("filename") or "output.csv"

        return send_file(
            io.BytesIO(blob.read()),
            as_attachment=True,
            download_name=filename,
            mimetype=blob.content_type or "application/octet-stream",
        )
    except FileNotFoundError as exc:
        return json_error(str(exc), 404)
    except Exception as exc:
        return json_error(str(exc), 500)

@app.delete("/file/<file_id>")
@app.delete("/delete-file/<file_id>")
def delete_file(file_id: str):
    try:
        _id = validate_object_id(file_id)

        doc = meta.find_one({"file_id": _id}) or cleaned_meta.find_one({"file_id": _id})
        if not doc:
            return json_error("File not found", 404)

        delete_gridfs_file(_id)
        meta.delete_many({"file_id": _id})
        cleaned_meta.delete_many({"file_id": _id})

        return jsonify({"status": "deleted", "file_id": file_id})
    except Exception as exc:
        return json_error(str(exc), 500)

@app.get("/files")
def list_files():
    try:
        originals = []
        for doc in meta.find().sort("created_at", -1):
            originals.append(
                {
                    "file_id": str(doc["file_id"]),
                    "filename": doc.get("filename"),
                    "created_at": doc.get("created_at").isoformat() if doc.get("created_at") else None,
                    "expires_at": doc.get("expires_at").isoformat() if doc.get("expires_at") else None,
                    "kind": doc.get("kind"),
                }
            )

        cleaned = []
        for doc in cleaned_meta.find().sort("created_at", -1):
            cleaned.append(
                {
                    "file_id": str(doc["file_id"]),
                    "filename": doc.get("filename"),
                    "created_at": doc.get("created_at").isoformat() if doc.get("created_at") else None,
                    "expires_at": doc.get("expires_at").isoformat() if doc.get("expires_at") else None,
                    "kind": doc.get("kind"),
                }
            )

        return jsonify({"originals": originals, "cleaned": cleaned})
    except Exception as exc:
        return json_error(str(exc), 500)

@app.post("/purge-expired")
def purge_expired():
    try:
        cleanup_expired_files()
        return jsonify({"status": "ok"})
    except Exception as exc:
        return json_error(str(exc), 500)

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    start_reaper_once()
    app.run(host="0.0.0.0", port=PORT, debug=False, use_reloader=False)
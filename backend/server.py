"""Local HTTP API and static server for the React Prakriti interface."""

import base64
import contextlib
import csv
import io
import json
import os
import random
import shutil
import sys
import tempfile
import threading
import traceback
import webbrowser
import zipfile
from datetime import datetime
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from uuid import uuid4

from PIL import Image

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent if (BACKEND_DIR.parent / "frontend").exists() else BACKEND_DIR
MINI_DIR = (PROJECT_ROOT / "mini") if (PROJECT_ROOT / "mini").exists() else BACKEND_DIR
FRONTEND_DIST = (PROJECT_ROOT / "frontend" / "dist") if (PROJECT_ROOT / "frontend" / "dist").exists() else (BACKEND_DIR / "dist")

for path in [BACKEND_DIR, PROJECT_ROOT, MINI_DIR]:
    path_str = str(path)
    if path.exists() and path_str not in sys.path:
        sys.path.insert(0, path_str)

from classifier_service import (  # noqa: E402
    VIEW_ORDER,
    analyze_images,
    clear_model_cache,
    generate_user_id,
    load_model,
)
from unified_cv import validate_image_quality  # noqa: E402

MAX_BODY_BYTES = 200 * 1024 * 1024
JOBS = {}
DOCTOR_TOKENS = set()
STATE_LOCK = threading.RLock()
IMPORT_LOCK = threading.Lock()


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _decode_image(data_url):
    if not isinstance(data_url, str):
        raise ValueError("The uploaded image is missing or invalid.")
    encoded = data_url.split(",", 1)[-1]
    try:
        image_bytes = base64.b64decode(encoded, validate=True)
    except Exception as exc:
        raise ValueError("The uploaded image could not be read.") from exc
    if len(image_bytes) > 35 * 1024 * 1024:
        raise ValueError("Each image must be smaller than 35 MB.")
    try:
        with Image.open(io.BytesIO(image_bytes)) as source:
            source.load()
            if source.format not in {"JPEG", "PNG"}:
                raise ValueError("Please use a JPG or PNG image.")
            return source.convert("RGB")
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError("The uploaded image could not be decoded.") from exc


def _image_data_url(image):
    if image is None:
        return None
    output = io.BytesIO()
    image.convert("RGB").save(output, format="JPEG", quality=84, optimize=True)
    encoded = base64.b64encode(output.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _new_job(kind):
    job_id = uuid4().hex
    job = {
        "id": job_id,
        "kind": kind,
        "state": "running",
        "progress": 0,
        "message": "Preparing…",
        "result": None,
        "error": None,
        "logs": "",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "temp_dir": None,
        "subjects": None,
        "lock": threading.Lock(),
    }
    with STATE_LOCK:
        JOBS[job_id] = job
    return job_id


def _update_job(job_id, **changes):
    with STATE_LOCK:
        job = JOBS.get(job_id)
        if job is not None:
            job.update(changes)


def _public_job(job_id):
    with STATE_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            return None
        fields = ("id", "kind", "state", "progress", "message", "result", "error", "logs", "created_at")
        return {key: _json_safe(job.get(key)) for key in fields}


def _analyze_request(payload):
    gender = str(payload.get("gender", "")).strip().lower()
    if gender not in {"male", "female"}:
        raise ValueError("Choose a gender before continuing.")
    try:
        age = int(payload.get("age"))
    except (TypeError, ValueError) as exc:
        raise ValueError("Enter an age between 1 and 120.") from exc
    if age < 1 or age > 120:
        raise ValueError("Enter an age between 1 and 120.")

    incoming_images = payload.get("images") or {}
    missing = [key for key in VIEW_ORDER if not incoming_images.get(key)]
    if missing:
        raise ValueError("Add all four hand photos before starting the analysis.")
    images = {key: _decode_image(incoming_images[key]) for key in VIEW_ORDER}
    quality = {key: validate_image_quality(images[key]) for key in VIEW_ORDER}
    result = analyze_images(images, age, gender, payload.get("observations") or {})

    user_id = generate_user_id()
    probabilities = result["probabilities"]
    ranked = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)

    from supabasehelper import upload_complete_analysis

    save_result = upload_complete_analysis(
        user_id=user_id,
        gender=gender,
        images_dict=images,
        probabilities={
            "Vata": probabilities.get("V", 0.0),
            "Pitta": probabilities.get("P", 0.0),
            "Kapha": probabilities.get("K", 0.0),
        },
        final_dosha=f"{result['dominant_name']} ({result['dominant_class']})",
        confidence=result["confidence"],
        age=age,
        top_class=result["dominant_class"],
        secondary_class=ranked[1][0] if len(ranked) > 1 else None,
        effective_markers=result["effective_markers"],
        is_doctor_mode=result["practitioner_mode"],
    )

    return {
        "analysis": _json_safe(result),
        "quality": _json_safe(quality),
        "save": {
            "success": bool(save_result.get("success")),
            "record_id": save_result.get("record_id"),
            "errors": save_result.get("errors", []),
            "user_id": user_id,
        },
    }


def _validate_image(payload):
    image = _decode_image(payload.get("image"))
    return _json_safe(validate_image_quality(image))


def _safe_extract_zip(zip_path, destination):
    root = destination.resolve()
    with zipfile.ZipFile(zip_path, "r") as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if target != root and root not in target.parents:
                raise ValueError("This ZIP contains a file path outside the archive folder.")
        archive.extractall(destination)


def _import_result_row(subject, index):
    return {
        "no": index + 1,
        "name": subject.get("name", "Unknown"),
        "age": subject.get("age", "Unknown"),
        "gender": subject.get("gender", "Unknown"),
        "skin_color": subject.get("skin_color", "Medium"),
        "skin_texture": subject.get("skin_texture") or subject.get("skin_type", "Smooth"),
        "palm_color": subject.get("palm_color", "PinkWhite"),
        "hair": f"{subject.get('hair_thickness', 'Thin')} ({subject.get('hair_color', 'Brown')})",
        "veins": subject.get("vein_visibility", "Low"),
        "mole": subject.get("mole", "No"),
        "predicted_dosha": subject.get("predicted_dosha", "Unknown"),
        "confidence": float(subject.get("confidence", 0.0) or 0.0),
        "kapha_prob": float(subject.get("kapha_prob", 0.0) or 0.0),
        "pitta_prob": float(subject.get("pitta_prob", 0.0) or 0.0),
        "vata_prob": float(subject.get("vata_prob", 0.0) or 0.0),
        "needs_review": bool(subject.get("needs_review", False)),
        "review_reasons": subject.get("review_reasons", []),
    }


def _run_import(job_id, archive_bytes, auto_append):
    from whatsapp_pipeline.pipeline import WhatsAppAIClassifierPipeline
    from whatsapp_pipeline.exporter import append_to_source_dataset
    from supabasehelper import get_gemini_api_key, get_groq_api_key

    temp_dir = Path(tempfile.mkdtemp(prefix="whatsapp_import_"))
    zip_path = temp_dir / "whatsapp_export.zip"
    output_excel = temp_dir / "whatsapp_batch_results.xlsx"
    zip_path.write_bytes(archive_bytes)
    _update_job(job_id, temp_dir=str(temp_dir), message="Checking the ZIP archive…", progress=1)

    old_groq_key = os.environ.get("GROQ_API_KEY")
    old_gemini_key = os.environ.get("GEMINI_API_KEY")
    logs = io.StringIO()
    try:
        groq_key = get_groq_api_key()
        gemini_key = get_gemini_api_key()
        os.environ["GROQ_API_KEY"] = groq_key or old_groq_key or ""
        os.environ["GEMINI_API_KEY"] = gemini_key or old_gemini_key or ""

        _safe_extract_zip(zip_path, temp_dir / "archive")
        extracted_dir = temp_dir / "archive"
        extracted_dir.mkdir(exist_ok=True)
        chat_files = list(extracted_dir.glob("*.txt"))
        if not chat_files:
            raise FileNotFoundError("Could not find a chat text file at the top level of the WhatsApp export ZIP.")
        chat_path = chat_files[0]
        shutil.copy2(chat_path, PROJECT_ROOT / "scratch_uploaded_chat.txt")

        def progress(percent, message):
            _update_job(
                job_id,
                progress=max(0, min(100, int(percent))),
                message=str(message),
            )

        with IMPORT_LOCK, contextlib.redirect_stdout(logs):
            pipeline = WhatsAppAIClassifierPipeline(confidence_threshold=0.0)
            subjects = pipeline.process_chat_export(
                chat_txt_path=chat_path,
                media_dir_path=extracted_dir,
                output_excel_path=output_excel,
                progress_callback=progress,
            )
            appended_ids = []
            if auto_append:
                progress(98, "Saving participants to the master dataset and cloud…")
                appended_ids = append_to_source_dataset(
                    subjects,
                    xlsx_path=PROJECT_ROOT / "dataset" / "palm_dataset_26 June 26.xlsx",
                    wrist_dataset_dir=PROJECT_ROOT / "dataset" / "Wrist_Dataset",
                ) or []
            else:
                print("\n[Safe Mode] Master dataset append skipped. Results are ready for review.")

        rows = [_import_result_row(subject, index) for index, subject in enumerate(subjects)]
        _update_job(
            job_id,
            state="complete",
            progress=100,
            message="Import complete.",
            result={
                "total": len(subjects),
                "saved_count": len(appended_ids) if auto_append else 0,
                "saved_ids": appended_ids,
                "auto_append": bool(auto_append),
                "rows": rows,
            },
            logs=logs.getvalue(),
            subjects=subjects if not auto_append else None,
            temp_dir=str(temp_dir) if not auto_append else None,
        )
        if auto_append:
            shutil.rmtree(temp_dir, ignore_errors=True)
    except Exception as exc:
        _update_job(
            job_id,
            state="error",
            message="The import could not be completed.",
            error=str(exc),
            logs=logs.getvalue() or traceback.format_exc(),
            subjects=None,
            temp_dir=None,
        )
        shutil.rmtree(temp_dir, ignore_errors=True)
    finally:
        if old_groq_key is None:
            os.environ.pop("GROQ_API_KEY", None)
        else:
            os.environ["GROQ_API_KEY"] = old_groq_key
        if old_gemini_key is None:
            os.environ.pop("GEMINI_API_KEY", None)
        else:
            os.environ["GEMINI_API_KEY"] = old_gemini_key


def _commit_import(job_id):
    with STATE_LOCK:
        job = JOBS.get(job_id)
        if not job or job.get("kind") != "whatsapp" or job.get("state") != "complete":
            raise ValueError("This completed batch is no longer available to save.")
        if job.get("result", {}).get("auto_append"):
            return job["result"]
        subjects = job.get("subjects")
        if subjects is None:
            raise ValueError("This batch has already been cleared.")

    from whatsapp_pipeline.exporter import append_to_source_dataset

    appended_ids = append_to_source_dataset(
        subjects,
        xlsx_path=PROJECT_ROOT / "dataset" / "palm_dataset_26 June 26.xlsx",
        wrist_dataset_dir=PROJECT_ROOT / "dataset" / "Wrist_Dataset",
    ) or []
    with STATE_LOCK:
        job = JOBS[job_id]
        job["subjects"] = None
        job["result"]["saved_count"] = len(appended_ids)
        job["result"]["saved_ids"] = appended_ids
        job["result"]["auto_append"] = True
        temp_dir = job.get("temp_dir")
        job["temp_dir"] = None
    if temp_dir:
        shutil.rmtree(temp_dir, ignore_errors=True)
    return job["result"]


def _run_retraining(job_id):
    from doctor_portal import retrain_model_pipeline

    def progress(fraction, message):
        _update_job(
            job_id,
            progress=max(0, min(100, int(float(fraction) * 100))),
            message=str(message),
        )

    try:
        result = retrain_model_pipeline(progress_callback=progress)
        if result.get("success"):
            clear_model_cache()
        _update_job(
            job_id,
            state="complete",
            progress=100 if result.get("success") else JOBS[job_id]["progress"],
            message="Model training complete." if result.get("success") else "Training needs attention.",
            result=_json_safe(result),
            error=None if result.get("success") else result.get("error"),
        )
    except Exception as exc:
        _update_job(
            job_id,
            state="error",
            message="Model training failed.",
            error=str(exc),
            logs=traceback.format_exc(),
        )


class AppHandler(SimpleHTTPRequestHandler):
    server_version = "PrakritiLocal/1.0"

    def __init__(self, *args, **kwargs):
        dist_dir = FRONTEND_DIST if FRONTEND_DIST.exists() else BACKEND_DIR
        super().__init__(*args, directory=str(dist_dir), **kwargs)

    def log_message(self, fmt, *args):
        print(f"[Prakriti] {self.address_string()} - {fmt % args}", file=sys.stderr)

    def _send_cors_headers(self):
        cors_origin = os.environ.get("CORS_ORIGIN", "*")
        self.send_header("Access-Control-Allow-Origin", cors_origin)
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS, PUT, DELETE")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization, X-Doctor-Token, X-Auto-Append, X-Upload-Filename, Accept, Origin")
        self.send_header("Access-Control-Max-Age", "86400")

    def _send_json(self, status, payload):
        raw = json.dumps(_json_safe(payload), ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self._send_cors_headers()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def _send_bytes(self, status, content_type, body, filename=None):
        self.send_response(status)
        self._send_cors_headers()
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        if filename:
            self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self):
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise ValueError("Invalid request length.") from exc
        if length > MAX_BODY_BYTES:
            raise OverflowError("This upload is too large for the local server.")
        raw = self.rfile.read(length)
        try:
            return json.loads(raw.decode("utf-8"))
        except Exception as exc:
            raise ValueError("The request data could not be read.") from exc

    def _doctor_allowed(self):
        token = self.headers.get("X-Doctor-Token", "")
        with STATE_LOCK:
            return bool(token) and token in DOCTOR_TOKENS

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/api/status":
            try:
                load_model()
                model_ready = True
                model_error = None
            except Exception as exc:
                model_ready = False
                model_error = str(exc)
            try:
                from supabasehelper import supabase_client

                cloud_ready = supabase_client is not None
            except Exception:
                cloud_ready = False
            return self._send_json(200, {
                "ok": True,
                "model_ready": model_ready,
                "model_error": model_error,
                "cloud_ready": cloud_ready,
            })

        if path.startswith("/api/jobs/"):
            job_id = path.rsplit("/", 1)[-1]
            job = _public_job(job_id)
            if not job:
                return self._send_json(404, {"error": "This task is no longer available."})
            return self._send_json(200, job)

        if path == "/api/doctor/summary":
            if not self._doctor_allowed():
                return self._send_json(401, {"error": "Please sign in to the practitioner workspace."})
            from doctor_portal import get_queue_summary

            pending, verified, duplicates = get_queue_summary()
            return self._send_json(200, {
                "pending": pending,
                "verified": verified,
                "duplicates": duplicates,
                "active_model": "Unified RBF-SVM",
            })

        if path == "/api/doctor/subject":
            if not self._doctor_allowed():
                return self._send_json(401, {"error": "Please sign in to the practitioner workspace."})
            subject_id = parse_qs(parsed.query).get("id", [""])[0]
            if not subject_id:
                return self._send_json(400, {"error": "Choose a participant to review."})
            from doctor_portal import get_subject_data

            subject = get_subject_data(subject_id)
            if subject.get("row_idx") is None:
                return self._send_json(404, {"error": "The selected participant could not be found."})
            images = {}
            for key, image in subject.get("images", {}).items():
                images[key] = _image_data_url(image)
                if image is not None:
                    image.close()
            subject["images"] = images
            return self._send_json(200, subject)

        if path.startswith("/api/"):
            return self._send_json(404, {"error": "This API address is not available."})

        if not (FRONTEND_DIST / "index.html").exists():
            status_html = """<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"><title>Prakriti API Backend</title>
<style>body{font-family:system-ui,-apple-system,sans-serif;background:#0d1117;color:#c9d1d9;display:flex;align-items:center;justify-content:center;min-height:100vh;margin:0;padding:1rem;}
.card{background:#161b22;padding:2rem 2.5rem;border-radius:12px;border:1px solid #30363d;max-width:540px;text-align:center;box-shadow:0 8px 24px rgba(0,0,0,0.4);}
h1{color:#58a6ff;font-size:1.6rem;margin-top:0;}
p{line-height:1.6;font-size:0.95rem;color:#8b949e;}
code{background:#21262d;color:#79c0ff;padding:0.2rem 0.5rem;border-radius:6px;font-size:0.9rem;}
a{color:#58a6ff;text-decoration:none;font-weight:600;}
a:hover{text-decoration:underline;}</style></head>
<body><div class="card">
<h1>🌿 Prakriti API Backend is Live</h1>
<p>The backend Python service is operational and ready to accept requests.</p>
<p>Service Health: <a href="/api/status"><code>/api/status</code></a></p>
<p>Connect your Netlify frontend by configuring the <code>VITE_API_BASE_URL</code> environment variable.</p>
</div></body></html>"""
            return self._send_bytes(200, "text/html; charset=utf-8", status_html.encode("utf-8"))
        return super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path
        try:
            if path == "/api/validate-image":
                payload = self._read_json()
                return self._send_json(200, _validate_image(payload))

            if path == "/api/analyze":
                payload = self._read_json()
                return self._send_json(200, _analyze_request(payload))

            if path == "/api/whatsapp/import":
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0:
                    return self._send_json(400, {"error": "Choose a WhatsApp ZIP file first."})
                if length > MAX_BODY_BYTES:
                    return self._send_json(413, {"error": "This ZIP is too large for the local server."})
                archive_bytes = self.rfile.read(length)
                auto_append = self.headers.get("X-Auto-Append", "true").lower() == "true"
                job_id = _new_job("whatsapp")
                threading.Thread(
                    target=_run_import,
                    args=(job_id, archive_bytes, auto_append),
                    daemon=True,
                ).start()
                return self._send_json(202, {"job_id": job_id})

            if path == "/api/whatsapp/commit":
                payload = self._read_json()
                result = _commit_import(str(payload.get("job_id", "")))
                return self._send_json(200, result)

            if path == "/api/doctor/login":
                payload = self._read_json()
                pin = str(payload.get("pin", ""))
                if pin not in {"1234", "doctor", "ayurveda", ""}:
                    return self._send_json(401, {"error": "That access code was not recognized."})
                token = uuid4().hex
                with STATE_LOCK:
                    DOCTOR_TOKENS.add(token)
                return self._send_json(200, {"token": token})

            if path == "/api/doctor/logout":
                token = self.headers.get("X-Doctor-Token", "")
                with STATE_LOCK:
                    DOCTOR_TOKENS.discard(token)
                return self._send_json(200, {"ok": True})

            if path == "/api/doctor/verify":
                if not self._doctor_allowed():
                    return self._send_json(401, {"error": "Please sign in to the practitioner workspace."})
                payload = self._read_json()
                subject_id = str(payload.get("subject_id", ""))
                dosha = str(payload.get("dosha", ""))
                features = payload.get("features") or {}
                required = {
                    "vein_visibility", "mole", "skin_texture", "skin_color",
                    "palm_color", "hair_thickness", "hair_color",
                }
                if not subject_id or dosha not in {"K-P", "P-K", "K-V", "V-K", "P-V", "V-P", "K", "P", "V"}:
                    return self._send_json(400, {"error": "Select a participant and a verified Prakriti."})
                if not required.issubset(features):
                    return self._send_json(400, {"error": "Complete all seven clinical features before saving."})
                from doctor_portal import save_doctor_verification

                save_doctor_verification(subject_id, dosha, features, str(payload.get("notes", "")))
                return self._send_json(200, {"ok": True, "subject_id": subject_id, "dosha": dosha})

            if path == "/api/doctor/retrain":
                if not self._doctor_allowed():
                    return self._send_json(401, {"error": "Please sign in to the practitioner workspace."})
                job_id = _new_job("retrain")
                threading.Thread(target=_run_retraining, args=(job_id,), daemon=True).start()
                return self._send_json(202, {"job_id": job_id})

            return self._send_json(404, {"error": "This API address is not available."})
        except OverflowError as exc:
            return self._send_json(413, {"error": str(exc)})
        except ValueError as exc:
            return self._send_json(400, {"error": str(exc)})
        except Exception as exc:
            return self._send_json(500, {"error": str(exc) or "Something went wrong."})

    def do_OPTIONS(self):
        self.send_response(204)
        self._send_cors_headers()
        self.send_header("Content-Length", "0")
        self.end_headers()


def main():
    import argparse

    default_host = os.environ.get("HOST", "0.0.0.0" if ("PORT" in os.environ or "RENDER" in os.environ) else "127.0.0.1")
    default_port = int(os.environ.get("PORT", 8501))

    parser = argparse.ArgumentParser(description="Run the Prakriti React app and local API.")
    parser.add_argument("--host", default=default_host)
    parser.add_argument("--port", type=int, default=default_port)
    parser.add_argument("--api-only", action="store_true", help="Run the API for the Vite development server.")
    parser.add_argument("--open-browser", action="store_true", help="Open the local app in the default browser.")
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), AppHandler)
    mode = "API for React development" if args.api_only else "React app and local API"
    print(f"{mode} running at http://{args.host}:{args.port}")
    if args.open_browser:
        threading.Timer(1.0, webbrowser.open, args=(f"http://{args.host}:{args.port}",)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping the Prakriti app.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

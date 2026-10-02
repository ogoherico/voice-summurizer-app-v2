#!/usr/bin/env python3
"""
app.py
------
Flask web interface for the voice file summarizer.

Upload an audio file -> processing runs in a background thread so the
browser never blocks -> the results page polls for progress and shows the
transcript, summary, and extracted action items when done. Every job is
recorded in jobs.sqlite3 (see jobs_db.py), so past results stay browsable
in /history even after the page is closed.

Run:
    python app.py

Then open http://127.0.0.1:5000 in your browser.

Security notes (see README for full detail):
  - Debug mode defaults OFF. Set FLASK_DEBUG=1 only for local development.
  - FLASK_SECRET_KEY should be set explicitly in any long-running/shared
    deployment; if unset, a random key is generated per process start.
  - Uploaded files are validated by both extension AND actual content
    (magic-byte sniffing), not just the filename.
  - CSRF protection is enabled on all forms via Flask-WTF.
  - Basic rate limiting is applied to the expensive/abusable endpoints.
  - Old uploads/results are automatically swept on a schedule (see
    RESULT_RETENTION_HOURS below); job HISTORY (text only, in the database)
    is kept separately and persists unless HISTORY_RETENTION_DAYS is set.
"""

import os
import re
import secrets
import subprocess
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # loads variables from a local .env file, if present

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    send_from_directory,
    jsonify,
    flash,
)
from flask_wtf import CSRFProtect
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.utils import secure_filename

import jobs_db as db
from voice_summarizer import transcribe_and_translate, summarize_auto, extract_action_items
from exporters import build_docx, build_pdf, send_summary_email

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
RESULTS_DIR = BASE_DIR / "results"
UPLOAD_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
db.init_db()

ALLOWED_EXTENSIONS = {"mp3", "wav", "m4a", "ogg", "flac", "webm", "mp4", "aac"}
# MIME/type prefixes we consider valid audio content once we sniff the
# actual file bytes (not just trust the extension). Browser-recorded
# clips often come through as webm/mp4 containers holding an audio track.
ALLOWED_CONTENT_PREFIXES = ("audio/", "video/webm", "video/mp4")

MAX_CONTENT_LENGTH = int(os.environ.get("MAX_UPLOAD_MB", "200")) * 1024 * 1024
MAX_AUDIO_SECONDS = int(os.environ.get("MAX_AUDIO_SECONDS", "3600"))  # 1 hour default
RESULT_RETENTION_HOURS = float(os.environ.get("RESULT_RETENTION_HOURS", "24"))
HISTORY_RETENTION_DAYS = os.environ.get("HISTORY_RETENTION_DAYS")  # unset = keep forever

JOB_ID_RE = re.compile(r"[0-9a-f]{12}")
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

_secret_key = os.environ.get("FLASK_SECRET_KEY")
if not _secret_key:
    # No key configured: generate one for this process only. Fine for a
    # single-worker local/dev run; set FLASK_SECRET_KEY explicitly for any
    # deployment with multiple worker processes or that needs sessions/CSRF
    # tokens to survive a restart.
    _secret_key = secrets.token_hex(32)
    print(
        "[security] FLASK_SECRET_KEY not set — using a random key for this "
        "process. Set FLASK_SECRET_KEY in your environment for production use.",
        flush=True,
    )
app.secret_key = _secret_key

csrf = CSRFProtect(app)
limiter = Limiter(get_remote_address, app=app, default_limits=["60 per hour"])


@app.template_filter("datetimeformat")
def datetimeformat(value):
    try:
        return datetime.fromtimestamp(value).strftime("%Y-%m-%d %H:%M")
    except (TypeError, ValueError, OSError):
        return "—"


# --------------------------------------------------------------------------- #
# Upload validation helpers
# --------------------------------------------------------------------------- #
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def sniff_is_audio(path: Path) -> bool:
    """
    Checks the file's actual content (not just its extension) to confirm
    it looks like audio/video, using magic-byte detection. Returns True
    if we can't form an opinion (unknown format) rather than blocking a
    legitimate file we simply don't recognize — extension + size limits
    are the primary defense; this catches the obvious "renamed .exe to
    .mp3" case.
    """
    try:
        import filetype

        kind = filetype.guess(str(path))
        if kind is None:
            return True  # unrecognized by this lightweight library; don't block
        return kind.mime.startswith(ALLOWED_CONTENT_PREFIXES)
    except Exception:
        # If the detection library itself fails for any reason, don't let
        # that break uploads — fall back to trusting the extension check.
        return True


def probe_duration_seconds(path: Path) -> float | None:
    """Uses ffprobe (ships with ffmpeg) to read audio duration without
    fully decoding the file. Returns None if ffprobe isn't available or
    the probe fails, in which case duration simply isn't enforced."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "csv=p=0",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        return float(result.stdout.strip())
    except Exception:
        return None


def cleanup_old_files() -> None:
    """Deletes uploads/results older than RESULT_RETENTION_HOURS. This only
    touches files on disk (raw audio, generated .docx/.pdf/.txt) — job
    HISTORY in the database is unaffected, and downloads are regenerated
    on demand from the stored text if the file is gone (see /download)."""
    cutoff = time.time() - (RESULT_RETENTION_HOURS * 3600)
    for directory in (UPLOAD_DIR, RESULTS_DIR):
        for f in directory.glob("*"):
            try:
                if f.is_file() and f.stat().st_mtime < cutoff and f.name != ".gitkeep":
                    f.unlink()
            except OSError:
                pass  # best-effort cleanup; don't let this break a request
    if HISTORY_RETENTION_DAYS:
        try:
            db.delete_old_jobs(float(HISTORY_RETENTION_DAYS))
        except ValueError:
            pass


def build_fallback_note(engine: str, failures: list[tuple[str, str]]) -> str | None:
    if not failures:
        return None
    skipped = ", ".join(label for label, _ in failures)
    used = "the local model" if engine == "Local model" else engine
    return f"{skipped} was unavailable, so {used} was used instead."


# --------------------------------------------------------------------------- #
# Background worker
# --------------------------------------------------------------------------- #
def process_job(job_id: str, audio_path: Path) -> None:
    """
    Runs the full pipeline for one job on a background thread. All progress
    and results are written to the database (jobs_db) rather than returned,
    since nothing is waiting synchronously for this function — the browser
    finds out what happened by polling GET /job/<job_id>/status.
    """
    try:
        db.update_job(job_id, status="processing", progress="Transcribing audio...")
        english_text, lang_code, lang_prob, whisper_model = transcribe_and_translate(
            str(audio_path)
        )

        if not english_text:
            db.update_job(job_id, status="error", error="No speech was detected in that file.")
            return

        db.update_job(job_id, progress="Summarizing...")
        summary, engine, failures = summarize_auto(english_text)
        fallback_note = build_fallback_note(engine, failures)

        db.update_job(job_id, progress="Finding action items...")
        action_items, ai_engine, ai_note = extract_action_items(english_text)

        db.update_job(job_id, progress="Preparing downloads...")
        lang_prob_str = f"{lang_prob:.0%}"
        build_docx(summary, english_text, lang_code, lang_prob_str, str(RESULTS_DIR / f"{job_id}_summary.docx"))
        build_pdf(summary, english_text, lang_code, lang_prob_str, str(RESULTS_DIR / f"{job_id}_summary.pdf"))
        (RESULTS_DIR / f"{job_id}_transcript.txt").write_text(english_text, encoding="utf-8")
        (RESULTS_DIR / f"{job_id}_summary.txt").write_text(summary, encoding="utf-8")

        db.update_job(
            job_id,
            status="done",
            progress="Done.",
            lang_code=lang_code,
            lang_prob=lang_prob_str,
            whisper_model=whisper_model,
            engine=engine,
            fallback_note=fallback_note,
            summary=summary,
            transcript=english_text,
            action_items=action_items,
            action_items_note=ai_note,
        )
    except Exception as exc:  # noqa: BLE001 - record it; the page polls and shows it
        db.update_job(job_id, status="error", error=str(exc))
    finally:
        try:
            audio_path.unlink(missing_ok=True)
        except OSError:
            pass


def ensure_downloadable(job: dict, kind: str) -> Path | None:
    """
    Returns the path to the requested export, regenerating it from the
    job's stored summary/transcript text if the on-disk file was removed
    by cleanup_old_files() — keeps old /history entries downloadable
    without having to keep every generated file around forever.
    """
    job_id = job["id"]
    paths = {
        "transcript": (RESULTS_DIR / f"{job_id}_transcript.txt", "transcript"),
        "summary": (RESULTS_DIR / f"{job_id}_summary.txt", "summary"),
        "summary_docx": (RESULTS_DIR / f"{job_id}_summary.docx", "docx"),
        "summary_pdf": (RESULTS_DIR / f"{job_id}_summary.pdf", "pdf"),
    }
    if kind not in paths:
        return None
    path, fmt = paths[kind]
    if path.exists():
        return path
    if not job.get("summary"):
        return None  # job never completed; nothing to regenerate from

    if fmt == "transcript":
        path.write_text(job["transcript"] or "", encoding="utf-8")
    elif fmt == "summary":
        path.write_text(job["summary"], encoding="utf-8")
    elif fmt == "docx":
        build_docx(job["summary"], job["transcript"] or "", job["lang_code"] or "?", job["lang_prob"] or "?", str(path))
    elif fmt == "pdf":
        build_pdf(job["summary"], job["transcript"] or "", job["lang_code"] or "?", job["lang_prob"] or "?", str(path))
    return path


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.route("/", methods=["GET"])
def index():
    cleanup_old_files()
    return render_template("index.html")


@app.route("/summarize", methods=["POST"])
@limiter.limit("10 per hour")
def summarize():
    if "audio_file" not in request.files:
        flash("No file was uploaded.")
        return redirect(url_for("index"))

    file = request.files["audio_file"]
    if file.filename == "":
        flash("No file was selected.")
        return redirect(url_for("index"))

    if not allowed_file(file.filename):
        flash(f"Unsupported file type. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}")
        return redirect(url_for("index"))

    job_id = uuid.uuid4().hex[:12]
    safe_name = secure_filename(file.filename)
    audio_path = UPLOAD_DIR / f"{job_id}_{safe_name}"
    file.save(audio_path)

    # Content-sniff: does this actually look like audio, regardless of extension?
    if not sniff_is_audio(audio_path):
        audio_path.unlink(missing_ok=True)
        flash("That file's content doesn't look like a supported audio format.")
        return redirect(url_for("index"))

    # Duration cap: protects against extremely long files tying up the server
    duration = probe_duration_seconds(audio_path)
    if duration is not None and duration > MAX_AUDIO_SECONDS:
        audio_path.unlink(missing_ok=True)
        flash(f"That file is too long ({duration/60:.1f} min). Max allowed is {MAX_AUDIO_SECONDS/60:.0f} min.")
        return redirect(url_for("index"))

    db.create_job(job_id, safe_name)
    threading.Thread(target=process_job, args=(job_id, audio_path), daemon=True).start()
    return redirect(url_for("job_page", job_id=job_id))


@app.route("/job/<job_id>")
def job_page(job_id):
    if not JOB_ID_RE.fullmatch(job_id):
        return "Not found", 404
    job = db.get_job(job_id)
    if job is None:
        flash("That job was not found — it may have been removed.")
        return redirect(url_for("index"))
    return render_template("job.html", job=job)


@app.route("/job/<job_id>/status")
@limiter.exempt  # polled every ~2s by the results page; the default 60/hour
                  # limit would cut off a job partway through for anything
                  # slower than ~2 minutes. It only reads existing status —
                  # no processing is triggered, so exempting it is safe.
def job_status(job_id):
    """Polled by the results page's JavaScript every couple of seconds."""
    if not JOB_ID_RE.fullmatch(job_id):
        return jsonify(error="not found"), 404
    job = db.get_job(job_id)
    if job is None:
        return jsonify(error="not found"), 404
    return jsonify(
        status=job["status"],
        progress=job["progress"],
        error=job["error"],
    )


@app.route("/history")
def history():
    jobs = db.list_jobs(limit=100)
    return render_template("history.html", jobs=jobs)


@app.route("/download/<job_id>/<kind>")
@limiter.limit("30 per hour")
def download(job_id, kind):
    if not JOB_ID_RE.fullmatch(job_id):
        return "Not found", 404
    job = db.get_job(job_id)
    if job is None:
        return "Not found", 404
    path = ensure_downloadable(job, kind)
    if path is None or not path.exists():
        return "Not found", 404
    return send_from_directory(path.parent, path.name, as_attachment=True)


@app.route("/email/<job_id>", methods=["POST"])
@limiter.limit("5 per hour")
def email_summary(job_id):
    if not JOB_ID_RE.fullmatch(job_id):
        return "Not found", 404

    to_address = request.form.get("email", "").strip()
    attach_format = request.form.get("attach_format", "docx")
    if attach_format not in {"docx", "pdf", "txt"}:
        attach_format = "docx"

    if not to_address or not EMAIL_RE.match(to_address):
        flash("Please enter a valid email address.")
        return redirect(url_for("job_page", job_id=job_id))

    job = db.get_job(job_id)
    if job is None or not job.get("summary"):
        flash("That result has expired or was not found. Please re-run the summarizer.")
        return redirect(url_for("index"))

    attachment = ensure_downloadable(job, f"summary_{attach_format}" if attach_format != "txt" else "summary")
    attachments = [str(attachment)] if attachment else []

    try:
        send_summary_email(
            to_address=to_address,
            subject="Your Voice File Summary",
            body=job["summary"],
            attachment_paths=attachments,
        )
        flash(f"Summary emailed to {to_address}.")
    except Exception as exc:  # noqa: BLE001 - surface the reason to the user
        flash(f"Could not send email: {exc}")

    return redirect(url_for("job_page", job_id=job_id))


if __name__ == "__main__":
    debug_mode = os.environ.get("FLASK_DEBUG", "0") == "1"
    # threaded=True lets the background job thread and the browser's status
    # polling requests run concurrently under Flask's own dev server.
    app.run(debug=debug_mode, host="127.0.0.1", port=5000, threaded=True)

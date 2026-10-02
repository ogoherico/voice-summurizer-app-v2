# Voice File Summarizer

Upload or point it at an audio file in **any spoken language** — it
transcribes the speech, translates it to English, and produces an English
summary. Comes with both a **Flask web interface** and a **command-line
tool**.

## How it works

1. **Whisper** (via `faster-whisper`) transcribes the audio and auto-detects
   the spoken language. Using Whisper's built-in `translate` task, it outputs
   the transcript directly in **English** — no separate translation step or
   language flag needed.
2. The English transcript is **summarized**. The app picks the engine itself,
   so the web page has no settings to choose from:
   - **Google Gemini** is tried first (needs `GOOGLE_API_KEY` in `.env`).
   - If Gemini isn't configured or fails (no internet, quota, retired model),
     it falls back to a **local** Hugging Face model (`facebook/bart-large-cnn`)
     that runs on your machine with no API key.
   - OpenAI can be added to the chain with `SUMMARY_BACKENDS` in `.env`.

   The **speech model** works the same way: the best Whisper model that's
   already downloaded is used (`small`, then `base`, then `tiny` by default).
   Download a bigger one once and the app starts using it automatically.

   The results page shows which engines were used, and tells you if a fallback
   happened.

   **Privacy:** while Gemini is in the chain, transcript text is sent to
   Google for summarizing. If your recordings are sensitive, set
   `SUMMARY_BACKENDS=local` in `.env` to keep everything on your machine.

## Setup

1. Install [ffmpeg](https://ffmpeg.org/download.html) (required for reading
   audio files):
   - macOS: `brew install ffmpeg`
   - Ubuntu/Debian: `sudo apt install ffmpeg`
   - Windows: download a build and add it to your PATH.

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Copy the example environment file and fill in what you need:
   ```bash
   cp .env.example .env
   ```
   Then edit `.env` — set `OPENAI_API_KEY` / `GOOGLE_API_KEY` if you'll use
   those backends, SMTP settings if you'll use email, and (recommended for
   anything beyond quick local testing) a `FLASK_SECRET_KEY`. The app loads
   `.env` automatically on startup. `.env` is gitignored, so these values
   never get committed.

## Option A: Web interface (Flask)

```bash
python app.py
```

Then open **http://127.0.0.1:5000** in your browser:

1. Choose an audio file (mp3, wav, m4a, ogg, flac, webm, mp4, aac).
2. Click "Transcribe & Summarize" (no settings to pick).
3. You're taken to a progress page immediately; processing happens in the
   background, so the browser never freezes, and the page updates itself
   when finished.
4. View the detected language, the summary, extracted **action items**, and
   the full transcript, and download the summary as **.txt, .docx (Word),
   or .pdf**, or the transcript as .txt.
5. Optionally, enter an email address and click **Send** to have the
   summary emailed with your chosen attachment format.
6. Visit **History** (top-right link) any time to revisit past jobs.

### Setting up email sending

Set these in your `.env` file (see step 3 above):

```
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=you@gmail.com
SMTP_PASSWORD=your-app-password
SMTP_FROM=you@gmail.com   # optional, defaults to SMTP_USER
```

**For Gmail specifically**, you can't use your normal account password —
you need an **App Password**:
1. Turn on 2-Step Verification on your Google Account (Security settings).
2. Go to Google Account → Security → App passwords.
3. Generate one for "Mail" and use that 16-character value as `SMTP_PASSWORD`.

Other providers (Outlook, Yahoo, a work SMTP server, etc.) work the same
way — just point `SMTP_HOST`/`SMTP_PORT` at their server and use the
appropriate login/password for that account.

If these variables aren't set, clicking "Send" will show an error saying
email isn't configured — everything else in the app still works fine.

Notes:
- Processing runs synchronously on the server — a long audio file with a
  bigger model will take a while, and the browser will just wait for the
  page to reload with results. For production use with long files, you'd
  want to move this to a background job queue (e.g. Celery + Redis) so the
  request doesn't time out; this app is intentionally simple for local/demo use.
- Uploaded audio files are deleted right after processing. Result text files
  are kept in `results/` so the download links keep working; clear that
  folder periodically if you're running this long-term.
- Set `FLASK_SECRET_KEY` in the environment for anything beyond local testing.

## Option B: Command line

```bash
python voice_summarizer.py meeting.mp3
```

Use a bigger, more accurate Whisper model:
```bash
python voice_summarizer.py meeting.mp3 --model medium
```

Use OpenAI for the summarization step:
```bash
python voice_summarizer.py meeting.mp3 --backend openai
```

Use Gemini for the summarization step:
```bash
python voice_summarizer.py meeting.mp3 --backend gemini
```

Save the full transcript and the summary to files:
```bash
python voice_summarizer.py meeting.mp3 \
    --transcript-out transcript.txt \
    --summary-out summary.txt
```

### CLI options

| Flag               | Description                                                        | Default        |
|--------------------|----------------------------------------------------------------------|----------------|
| `--model`          | Whisper model: `auto`, tiny, base, small, medium, large-v3          | `auto`         |
| `--backend`        | Summarizer: `auto`, `local`, `openai`, or `gemini`                   | `auto`         |
| `--openai-model`   | OpenAI model name (if `--backend openai`)                           | `gpt-4o-mini`  |
| `--gemini-model`   | Gemini model name (if `--backend gemini`)                            | `GEMINI_MODEL` from `.env`, else `gemini-3.8-flash` |
| `--transcript-out` | Path to save the full English transcript                            | (not saved)    |
| `--summary-out`    | Path to save just the summary                                       | (not saved)    |

## New in this version

- **Background processing.** Uploading no longer freezes the browser; you
  are redirected to a job page that shows live progress and refreshes
  automatically when done.
- **History.** Every job (done or failed) is listed at `/history`, linking
  back to its full results. Stored in `jobs.sqlite3` (a local file,
  gitignored; contains transcript/summary text, so do not share this file).
  Keeps everything by default; set `HISTORY_RETENTION_DAYS` in `.env` to
  prune it automatically after N days.
- **Action items.** Alongside the summary, the app extracts a list of
  concrete tasks/commitments/deadlines mentioned in the recording. This
  step needs Google Gemini or OpenAI configured (the local model is not
  reliable at structured extraction); if neither is set up, this section
  just says so rather than guessing.
- **Durable downloads.** Summary/transcript text is kept in the database,
  so a download from `/history` still works even after the original
  generated .docx/.pdf files were cleaned up by `RESULT_RETENTION_HOURS`;
  the file is regenerated on the spot from the stored text.

## Security

This app includes several protections out of the box, on by default:

| Protection | What it does |
|---|---|
| **Debug mode off by default** | `FLASK_DEBUG` defaults to `0`. Flask's debugger, if left on and exposed, lets anyone reaching the server run arbitrary Python. Only set `FLASK_DEBUG=1` for local development on your own machine. |
| **Secrets via `.env`, not hardcoded** | Credentials (API keys, SMTP login) load from `.env`, which is gitignored. Never hardcode real credentials into `.py` files, especially if you'll ever share or push this code. |
| **Random secret key if unset** | If `FLASK_SECRET_KEY` isn't set, a random one is generated per run. Fine for solo local use; set it explicitly for anything longer-running or multi-process, since sessions/CSRF tokens won't survive a restart otherwise. |
| **CSRF protection** | All forms (upload, email) include a CSRF token via Flask-WTF, blocking cross-site request forgery attempts. |
| **Rate limiting** | `/summarize` (10/hour), `/email` (5/hour), `/download` (30/hour), and a 60/hour default elsewhere — via Flask-Limiter — to stop one visitor from hammering the expensive endpoints or running up your API bill. |
| **Upload content validation** | Files are checked both by extension AND by sniffing actual file bytes (`filetype` library), so a renamed `.exe` claiming to be `.mp3` gets rejected rather than trusted on filename alone. |
| **Upload size + duration caps** | `MAX_UPLOAD_MB` (default 200MB) and `MAX_AUDIO_SECONDS` (default 1 hour) both configurable in `.env`. |
| **Automatic file retention/cleanup** | Uploaded audio and generated results older than `RESULT_RETENTION_HOURS` (default 24) are deleted automatically. Don't rely on this alone for sensitive data — see below. |
| **Path safety on downloads** | The `/download/<job_id>/...` route validates `job_id` matches the exact format the app generates, rather than trusting arbitrary user input in a file lookup. |

### What's still on you

- **This app has no login/authentication.** Anyone who can reach the URL can use it and see anyone else's results while they're within the retention window. Fine for personal/local use; add real auth (e.g. Flask-Login) before exposing this to other people.
- **Put it behind HTTPS** the moment it's reachable by anyone besides you on localhost — run it behind a reverse proxy (nginx, Caddy) with a TLS certificate (e.g. via Let's Encrypt), rather than serving plain HTTP.
- **Don't run this with Flask's built-in server in production** — use a real WSGI server (`gunicorn`, `waitress`) behind that reverse proxy. The dev server used by `app.run(...)` isn't designed to handle real traffic or hostile input robustly.
- **Rotate any credential that ever got typed into a chat, screen-shared, or committed to git history.** If in doubt, regenerate it.
- **Sensitive transcripts**: voice memos can contain very personal information. If this ever serves other people's data, consider encrypting `results/` at rest and shortening `RESULT_RETENTION_HOURS`.



- `tiny` / `base`: fast, decent for clear speech, less accurate on accents
  or background noise.
- `small` / `medium`: noticeably better accuracy, slower.
- `large-v3`: best accuracy, slowest, needs more RAM (a GPU helps a lot here).

The first time you run a given Whisper or Hugging Face model, it will be
downloaded and cached locally (a few hundred MB to a few GB depending on
size), so the first run is slower than subsequent ones.

## Project structure

```
voice-summarizer/
├── app.py                 # Flask web app (routes, background jobs)
├── voice_summarizer.py    # Core pipeline + CLI (transcription + summarization + action items)
├── exporters.py           # Word/PDF export + email sending
├── jobs_db.py             # SQLite-backed job store (history, progress, durable downloads)
├── jobs.sqlite3           # Created automatically on first run (gitignored)
├── templates/
│   ├── _header.html       # Shared header, included by every page
│   ├── index.html         # Upload page
│   ├── job.html           # Processing / results page (polls for progress)
│   └── history.html       # Past jobs list
├── static/
│   ├── style.css          # Shared styles for all pages
│   └── logo.png           # Header logo
├── .env.example           # Copy to .env and fill in your own values
├── .gitignore
├── requirements.txt
└── README.md
```

## Extending it

- **Background jobs**: move processing into a Celery/RQ task so uploads
  don't block the web request, and poll or use WebSockets for progress.
- **Batch processing**: loop over a folder of audio files with the CLI and
  write one summary file per input.
- **Speaker diarization**: add `pyannote.audio` if you need "who said what"
  before summarizing.
- **Deployment**: run behind `gunicorn`/`waitress` and a reverse proxy
  (nginx) instead of Flask's built-in dev server for anything beyond local use.

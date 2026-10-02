#!/usr/bin/env python3
"""
voice_summarizer.py
--------------------
AI-powered voice file summarizer.

Pipeline:
  1. Load an audio file (any common format: mp3, wav, m4a, ogg, flac, etc.)
  2. Transcribe + translate it to English using OpenAI's Whisper model
     (Whisper auto-detects the spoken language; no language flag needed).
  3. Summarize the resulting English text using one of:
       - a local Hugging Face summarization model (no API key, runs offline),
       - the OpenAI Chat API (needs OPENAI_API_KEY), or
       - the Google Gemini API (needs GOOGLE_API_KEY / GEMINI_API_KEY).

Usage:
    python voice_summarizer.py path/to/audio.mp3
    python voice_summarizer.py path/to/audio.mp3 --backend openai
    python voice_summarizer.py path/to/audio.mp3 --backend gemini
    python voice_summarizer.py path/to/audio.mp3 --model medium --backend local
    python voice_summarizer.py path/to/audio.mp3 --transcript-out transcript.txt

Requires ffmpeg to be installed on the system (used internally by faster-whisper).
"""

import argparse
import json
import os
import re
import sys
import textwrap
from pathlib import Path

try:  # let the CLI pick up the same .env file the web app uses
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

SUMMARY_SYSTEM_PROMPT = (
    "You summarize transcripts concisely and accurately in English. "
    "Produce a clear summary covering the main points, key facts, "
    "and any decisions or action items mentioned. Use short paragraphs "
    "or bullet points as appropriate."
)

# Gemini model names change often and old ones get shut down (gemini-2.0-flash
# stopped working on June 1, 2026). Override with GEMINI_MODEL in .env or the
# --gemini-model flag rather than editing code.
DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"


def clean_api_key(raw: str | None, var_names: str) -> str:
    """
    Normalizes an API key read from the environment and fails with a clear
    message if it can't be used. Stray spaces, line breaks (common with
    Windows line endings) or wrapping quotes in a .env value otherwise
    surface as confusing errors deep inside the API client, e.g.
    "Illegal header value" from Gemini's gRPC layer. The key itself is never
    included in the error message.
    """
    key = (raw or "").strip().strip("\"'").strip()
    if not key:
        raise RuntimeError(
            f"{var_names} is not set. Add it to your .env file, or run with "
            "--backend local."
        )
    if not key.isascii() or any(ch.isspace() or ord(ch) < 33 for ch in key):
        raise RuntimeError(
            f"{var_names} contains spaces, line breaks or hidden characters. "
            "Re-copy the key, paste it on a single line in .env with no quotes, "
            "and make sure no old value is still set in your terminal "
            "(PowerShell: Remove-Item Env:GOOGLE_API_KEY)."
        )
    return key


# --------------------------------------------------------------------------- #
# Step 1: Transcription + translation (any language -> English text)
# --------------------------------------------------------------------------- #
ALLOWED_WHISPER_MODELS = ("tiny", "base", "small", "medium", "large-v3")
# Preference order, best first. The app uses the best one that actually works
# on this machine and quietly falls back to the next. Override with the
# WHISPER_MODELS setting in .env, e.g. WHISPER_MODELS=medium,small,base,tiny
DEFAULT_WHISPER_CHAIN = "small,base,tiny"

ALLOWED_SUMMARY_BACKENDS = ("gemini", "openai", "local")
# Tried in order; the first one that succeeds wins. Override with
# SUMMARY_BACKENDS in .env, e.g. SUMMARY_BACKENDS=local to keep every
# transcript on this machine.
DEFAULT_SUMMARY_CHAIN = "gemini,local"
BACKEND_LABELS = {"gemini": "Google Gemini", "openai": "OpenAI", "local": "Local model"}


def _parse_chain(env_name: str, default: str, allowed: tuple[str, ...]) -> list[str]:
    """Reads a comma-separated preference list from the environment,
    ignoring blanks, duplicates and unknown names."""
    chain: list[str] = []
    for item in (os.environ.get(env_name) or default).split(","):
        item = item.strip().lower()
        if item in allowed and item not in chain:
            chain.append(item)
    return chain or [x for x in default.split(",")]


def _short(exc: Exception, limit: int = 140) -> str:
    text = " ".join(str(exc).split()) or type(exc).__name__
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _run_whisper(model, audio_path: str) -> tuple[str, str, float]:
    """Transcribes + translates with an already-loaded model. The segments
    come from a lazy generator, so any decoding/memory error surfaces while
    iterating here, which is why this must run inside the caller's try."""
    print(f"[1/2] Transcribing + translating '{audio_path}' ...", file=sys.stderr)
    segments, info = model.transcribe(
        audio_path,
        task="translate",   # <-- this is what gives us English output directly
        vad_filter=True,    # skips silence, improves accuracy on long files
    )
    pieces = []
    for seg in segments:
        pieces.append(seg.text.strip())
        print(f"    [{seg.start:6.1f}s -> {seg.end:6.1f}s] {seg.text.strip()}", file=sys.stderr)
    return " ".join(pieces).strip(), info.language, info.language_probability


def transcribe_and_translate(
    audio_path: str, model_size: str = "auto"
) -> tuple[str, str, float, str]:
    """
    Runs Whisper on the audio file with task="translate", which transcribes
    speech in ANY supported language and outputs English text directly.

    model_size="auto" (the default) walks the preference chain
    (WHISPER_MODELS, default "small,base,tiny"):
      pass 1: only models already downloaded to this machine, best first.
              This is instant and needs no network, so a model that hasn't
              been downloaded yet can never stall a request.
      pass 2: only if nothing was usable locally, allow downloading, best
              first, moving to the next model if a download or run fails.
    A model that loads but fails while transcribing (e.g. out of memory) is
    skipped and the next one is tried. Passing an explicit size such as
    "base" uses just that model.

    Returns: (english_text, language_code, language_probability, model_used)
    """
    from faster_whisper import WhisperModel

    if model_size == "auto":
        chain = _parse_chain("WHISPER_MODELS", DEFAULT_WHISPER_CHAIN, ALLOWED_WHISPER_MODELS)
    else:
        chain = [model_size]

    problems: list[str] = []
    run_failed: set[str] = set()

    for local_only in (True, False):
        for name in chain:
            if name in run_failed:
                continue
            try:
                # device="cpu" / compute_type="int8" keeps this runnable without a GPU.
                # Swap to device="cuda", compute_type="float16" if you have a GPU.
                model = WhisperModel(
                    name, device="cpu", compute_type="int8", local_files_only=local_only
                )
            except Exception as exc:  # noqa: BLE001
                if not local_only:  # "not downloaded yet" is expected in pass 1
                    problems.append(f"{name}: {_short(exc)}")
                continue

            print(f"[1/2] Using Whisper model '{name}'", file=sys.stderr)
            try:
                text, lang, prob = _run_whisper(model, audio_path)
                return text, lang, prob, name
            except Exception as exc:  # noqa: BLE001
                run_failed.add(name)
                problems.append(f"{name}: {_short(exc)}")
                print(f"[1/2] Model '{name}' failed, trying the next one.", file=sys.stderr)
            finally:
                del model

    raise RuntimeError(
        "Speech recognition failed with every available model. "
        + "; ".join(problems[-4:])
    )


# --------------------------------------------------------------------------- #
# Step 2a: Local summarization backend (no API key required)
# --------------------------------------------------------------------------- #
def summarize_local(text: str, max_chunk_chars: int = 3500) -> str:
    """
    Summarizes English text using a local Hugging Face model
    (facebook/bart-large-cnn), loaded directly via AutoTokenizer /
    AutoModelForSeq2SeqLM rather than pipeline("summarization", ...).

    This sidesteps a known issue on some transformers installs where the
    'summarization' pipeline alias isn't registered (you'd see an error
    like "Unknown task summarization, available tasks are [...]"), by
    calling the model's .generate() method directly instead.

    Long text is split into chunks, each chunk is summarized, and the
    chunk-summaries are combined and (if needed) summarized once more.
    """
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    model_name = "facebook/bart-large-cnn"
    print(f"[2/2] Loading local summarization model ({model_name}) ...", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def _summarize_one(chunk: str, max_length: int = 150, min_length: int = 30) -> str:
        inputs = tokenizer(
            [chunk],
            max_length=1024,
            truncation=True,
            return_tensors="pt",
        )
        summary_ids = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            num_beams=4,
            max_length=max_length,
            min_length=min_length,
            length_penalty=2.0,
            early_stopping=True,
        )
        return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    chunks = textwrap.wrap(text, max_chunk_chars, break_long_words=False)
    if not chunks:
        return ""

    print(f"[2/2] Summarizing {len(chunks)} chunk(s) ...", file=sys.stderr)
    chunk_summaries = []
    for i, chunk in enumerate(chunks, 1):
        chunk_summaries.append(_summarize_one(chunk))
        print(f"    chunk {i}/{len(chunks)} done", file=sys.stderr)

    combined = " ".join(chunk_summaries)

    # If we had multiple chunks, do one more pass to tighten the final summary.
    if len(chunk_summaries) > 1:
        return _summarize_one(combined, max_length=180, min_length=40)
    return combined


# --------------------------------------------------------------------------- #
# Step 2b: OpenAI summarization backend (better quality, needs API key)
# --------------------------------------------------------------------------- #
def summarize_openai(text: str, model: str = "gpt-4o-mini") -> str:
    """
    Summarizes English text using the OpenAI Chat Completions API.
    Requires the OPENAI_API_KEY environment variable to be set.
    """
    from openai import OpenAI

    api_key = clean_api_key(os.environ.get("OPENAI_API_KEY"), "OPENAI_API_KEY")

    print(f"[2/2] Summarizing with OpenAI model '{model}' ...", file=sys.stderr)
    client = OpenAI(api_key=api_key)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
            {"role": "user", "content": f"Summarize this transcript:\n\n{text}"},
        ],
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()


# --------------------------------------------------------------------------- #
# Step 2c: Gemini summarization backend (better quality, needs API key)
# --------------------------------------------------------------------------- #
def summarize_gemini(text: str, model: str | None = None) -> str:
    """
    Summarizes English text using Google's Gemini API via the current
    `google-genai` SDK (the older `google-generativeai` package is
    deprecated and no longer supported).

    Requires GOOGLE_API_KEY (or GEMINI_API_KEY) in the environment / .env.
    Get a key at https://aistudio.google.com/apikey
    The model defaults to DEFAULT_GEMINI_MODEL and can be overridden with the
    GEMINI_MODEL environment variable or the `model` argument.
    """
    from google import genai
    from google.genai import types

    api_key = clean_api_key(
        os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"),
        "GOOGLE_API_KEY (or GEMINI_API_KEY)",
    )
    model = model or os.environ.get("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL

    print(f"[2/2] Summarizing with Gemini model '{model}' ...", file=sys.stderr)
    # Request timeout (the SDK takes milliseconds) so a network problem fails
    # fast and the automatic fallback can take over instead of hanging.
    try:
        timeout_s = float(os.environ.get("GEMINI_TIMEOUT_SECONDS") or 30)
    except ValueError:
        timeout_s = 30.0
    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=int(timeout_s * 1000)),
    )
    response = client.models.generate_content(
        model=model,
        contents=f"Summarize this transcript:\n\n{text}",
        config=types.GenerateContentConfig(system_instruction=SUMMARY_SYSTEM_PROMPT),
    )

    summary = (response.text or "").strip()
    if not summary:
        raise RuntimeError(
            "Gemini returned an empty response (it may have been blocked by a "
            "safety filter). Try again or use another backend."
        )
    return summary


# --------------------------------------------------------------------------- #
# Automatic backend selection with fallback
# --------------------------------------------------------------------------- #
def summarize_auto(
    text: str, backends: list[str] | None = None
) -> tuple[str, str, list[tuple[str, str]]]:
    """
    Summarizes `text` with the first backend in the chain that works
    (SUMMARY_BACKENDS, default "gemini,local"). A backend that isn't set up
    (no API key) or that errors (network down, quota, retired model, ...)
    is skipped and the next one is tried.

    Returns: (summary, label_of_backend_used, [(label, short_reason), ...])
    where the list holds the backends that were tried and failed first.
    """
    chain = backends or _parse_chain(
        "SUMMARY_BACKENDS", DEFAULT_SUMMARY_CHAIN, ALLOWED_SUMMARY_BACKENDS
    )
    failures: list[tuple[str, str]] = []

    for name in chain:
        try:
            if name == "gemini":
                summary = summarize_gemini(text)
            elif name == "openai":
                summary = summarize_openai(text)
            else:
                summary = summarize_local(text)
            return summary, BACKEND_LABELS[name], failures
        except Exception as exc:  # noqa: BLE001 - fall through to the next backend
            failures.append((BACKEND_LABELS[name], _short(exc)))
            print(f"[2/2] {BACKEND_LABELS[name]} unavailable: {_short(exc)}", file=sys.stderr)

    raise RuntimeError(
        "Could not create a summary. "
        + "; ".join(f"{label}: {reason}" for label, reason in failures)
    )


# --------------------------------------------------------------------------- #
# Action-item extraction (structured: tasks, dates, names)
# --------------------------------------------------------------------------- #
ACTION_ITEM_PROMPT = (
    "Read this transcript and extract any concrete action items, commitments, "
    "deadlines, or follow-ups mentioned — things someone said they will do, "
    "need to do, or asked someone else to do. Ignore vague statements with "
    "no actual task attached.\n\n"
    "Respond with ONLY a JSON array (no prose, no markdown fences), where each "
    "item is an object with these keys:\n"
    '  "item": short description of the task (required)\n'
    '  "who": who is responsible, if stated, else null\n'
    '  "due": any date/deadline/timeframe mentioned, if stated, else null\n\n'
    "If there are no action items at all, respond with exactly: []\n\n"
    "Transcript:\n{text}"
)

# Rule-based extraction needs no API and is always available, but it only
# catches commonly-phrased commitments — it's a floor, not a replacement for
# the AI-backed extractors above it in the chain.
ACTION_ITEM_BACKENDS = ("gemini", "openai")


def _parse_action_items_json(raw: str) -> list[dict]:
    """
    Models sometimes wrap JSON in ```json fences or add a sentence before/
    after it despite instructions not to — this extracts just the [...] array
    and tolerates minor deviations rather than failing the whole response.
    """
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned)
        cleaned = re.sub(r"```\s*$", "", cleaned)
    start, end = cleaned.find("["), cleaned.rfind("]")
    if start == -1 or end == -1 or end < start:
        raise ValueError("Response did not contain a JSON array")
    items = json.loads(cleaned[start : end + 1])
    if not isinstance(items, list):
        raise ValueError("Parsed JSON was not a list")

    cleaned_items = []
    for it in items:
        if isinstance(it, str):
            it = {"item": it}
        if not isinstance(it, dict) or not it.get("item"):
            continue
        cleaned_items.append(
            {
                "item": str(it["item"]).strip(),
                "who": (str(it["who"]).strip() if it.get("who") else None),
                "due": (str(it["due"]).strip() if it.get("due") else None),
            }
        )
    return cleaned_items


def _extract_action_items_once(text: str, backend: str) -> list[dict]:
    prompt = ACTION_ITEM_PROMPT.format(text=text)
    if backend == "gemini":
        raw = summarize_gemini(prompt)
    else:
        raw = summarize_openai(prompt)
    return _parse_action_items_json(raw)


def extract_action_items(
    text: str, backends: list[str] | None = None
) -> tuple[list[dict], str | None, str | None]:
    """
    Pulls out a structured list of action items/commitments/deadlines from
    the transcript, using the same try-the-next-one-on-failure pattern as
    summarize_auto. This is intentionally AI-backed only (no local-model
    fallback): the local summarizer isn't reliable at structured extraction,
    and silently returning a bad guess would be worse than skipping it.

    Returns: (items, engine_used_or_None, skip_reason_or_None)
    skip_reason is set (and items is []) when no configured backend could
    do the extraction — e.g. neither Gemini nor OpenAI is set up.
    """
    chain = [
        b for b in (backends or _parse_chain(
            "SUMMARY_BACKENDS", DEFAULT_SUMMARY_CHAIN, ALLOWED_SUMMARY_BACKENDS
        ))
        if b in ACTION_ITEM_BACKENDS
    ]
    failures: list[str] = []

    for name in chain:
        try:
            items = _extract_action_items_once(text, name)
            return items, BACKEND_LABELS[name], None
        except Exception as exc:  # noqa: BLE001 - try the next backend
            failures.append(f"{BACKEND_LABELS[name]}: {_short(exc)}")
            print(f"[3/3] Action items via {BACKEND_LABELS[name]} failed: {_short(exc)}", file=sys.stderr)

    if not chain:
        return [], None, "Action item extraction needs Google Gemini or OpenAI configured."
    return [], None, "Could not extract action items (" + "; ".join(failures) + ")"


# --------------------------------------------------------------------------- #
# Main CLI
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(
        description="Transcribe a voice file in any language and summarize it in English."
    )
    parser.add_argument("audio_file", help="Path to the audio file (mp3, wav, m4a, ogg, flac, ...)")
    parser.add_argument(
        "--model",
        default="auto",
        choices=["auto", "tiny", "base", "small", "medium", "large-v3"],
        help=(
            "Whisper model size (bigger = more accurate, slower). Default: auto "
            "(best available model, falling back to smaller ones)"
        ),
    )
    parser.add_argument(
        "--backend",
        default="auto",
        choices=["auto", "local", "openai", "gemini"],
        help=(
            "Summarization backend. Default: auto (Gemini if configured, "
            "otherwise/if it fails the local model). 'local' needs no API key; "
            "'openai' needs OPENAI_API_KEY; 'gemini' needs GOOGLE_API_KEY."
        ),
    )
    parser.add_argument(
        "--openai-model",
        default="gpt-4o-mini",
        help="OpenAI model to use when --backend openai (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--gemini-model",
        default=None,
        help=(
            "Gemini model to use when --backend gemini (default: GEMINI_MODEL "
            f"from .env, else {DEFAULT_GEMINI_MODEL})"
        ),
    )
    parser.add_argument(
        "--transcript-out",
        default=None,
        help="Optional path to save the full English transcript as a .txt file",
    )
    parser.add_argument(
        "--summary-out",
        default=None,
        help="Optional path to save the summary as a .txt file",
    )
    args = parser.parse_args()

    audio_path = Path(args.audio_file)
    if not audio_path.exists():
        print(f"Error: file not found: {audio_path}", file=sys.stderr)
        sys.exit(1)

    # Step 1: transcribe + translate to English
    english_text, lang_code, lang_prob, whisper_used = transcribe_and_translate(
        str(audio_path), args.model
    )

    if not english_text:
        print("No speech detected in the audio file.", file=sys.stderr)
        sys.exit(1)

    if args.transcript_out:
        Path(args.transcript_out).write_text(english_text, encoding="utf-8")
        print(f"Full transcript saved to {args.transcript_out}", file=sys.stderr)

    # Step 2: summarize
    if args.backend == "auto":
        summary, engine, _failures = summarize_auto(english_text)
    elif args.backend == "openai":
        summary, engine = summarize_openai(english_text, model=args.openai_model), "OpenAI"
    elif args.backend == "gemini":
        summary, engine = summarize_gemini(english_text, model=args.gemini_model), "Google Gemini"
    else:
        summary, engine = summarize_local(english_text), "Local model"

    if args.summary_out:
        Path(args.summary_out).write_text(summary, encoding="utf-8")

    print("\n" + "=" * 70)
    print(f"Detected source language: {lang_code}  (confidence: {lang_prob:.0%})")
    print(f"Speech model: {whisper_used}   Summarized by: {engine}")
    print("=" * 70)
    print("SUMMARY (English):\n")
    print(summary)
    print("=" * 70)


if __name__ == "__main__":
    main()

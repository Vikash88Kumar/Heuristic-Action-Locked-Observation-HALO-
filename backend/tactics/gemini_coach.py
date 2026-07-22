"""
Calls the Gemini API to turn the pipeline's structured match data
(events, commentary, KPIs) plus sampled annotated-video frames into a
coach-style tactical report covering both teams equally.
"""
from __future__ import annotations

import json
import time

from google import genai
from google.genai import types

from . import config
from .prompt import COACH_ANALYSIS_PROMPT


class TacticsAnalysisError(RuntimeError):
    pass


def _condense_events(events: list[dict], max_events: int) -> list[dict]:
    """Down-sample the event log so the prompt stays a reasonable size
    on long matches, while keeping it evenly representative in time."""
    if len(events) <= max_events:
        return events
    step = len(events) / max_events
    out = []
    i = 0.0
    while len(out) < max_events and int(i) < len(events):
        out.append(events[int(i)])
        i += step
    return out


def _build_context_text(result: dict) -> str:
    kpis = result.get("kpis", {})
    events = _condense_events(result.get("events", []), config.TACTICS_MAX_EVENTS_IN_PROMPT)
    commentary = result.get("commentary", [])[:60]

    payload = {
        "duration_seconds": result.get("duration_sec"),
        "fps_source": result.get("fps_source"),
        "kpis": kpis,
        "event_log": events,
        "commentary_lines": commentary,
    }
    return json.dumps(payload, indent=2)


def _build_image_parts(frames: list[bytes]) -> list[types.Part]:
    return [types.Part.from_bytes(data=f, mime_type="image/jpeg") for f in frames]


def _parse_json_response(text: str) -> dict:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:].strip()
    return json.loads(cleaned)


def generate_coach_report(result: dict, frames: list[bytes]) -> dict:
    """
    result: the dict already returned by pipeline.VideoProcessor.process()
            -- {fps_source, duration_sec, commentary, events, kpis}
    frames: JPEG bytes sampled from the annotated output video
    """
    api_key = config.require_api_key()
    client = genai.Client(api_key=api_key)

    context_text = _build_context_text(result)
    prompt_text = (
        f"{COACH_ANALYSIS_PROMPT}\n\nStructured match data (JSON):\n{context_text}"
    )

    contents = [*_build_image_parts(frames), prompt_text]

    last_error = None
    for attempt in range(1, 4):
        try:
            response = client.models.generate_content(
                model=config.GEMINI_MODEL,
                contents=contents,
                config=types.GenerateContentConfig(response_mime_type="application/json"),
            )
            text = getattr(response, "text", "") or ""
            return _parse_json_response(text)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt < 3:
                time.sleep(2)

    raise TacticsAnalysisError(f"Gemini tactical analysis failed after 3 attempts: {last_error}")

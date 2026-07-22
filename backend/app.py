"""
HALO backend -- Flask app that serves the API the static index.html
frontend expects:

  Core pipeline (unchanged from the original HALO project):
    GET  /inputs
    POST /upload
    POST /process
    GET  /progress/<filename>
    GET  /result/<filename>
    GET  /outputs/<file>

  Tactics (new -- coach-style match analysis via Gemini, generated
  on demand, NOT automatically, so it never blocks or slows down the
  core detection/tracking pipeline):
    POST /tactics/<filename>            start tactical analysis job
    GET  /tactics/progress/<filename>   poll job status
    GET  /tactics/result/<filename>     JSON report + markdown text

Run with:
    python app.py

Then open static/index.html (or the deployed frontend) and point the
"Backend" field at this server's URL.
"""
import os
import threading
import traceback
import time

from flask import Flask, request, jsonify, send_from_directory

from models import load_jersey_cnn, load_ccnn_filter
from pipeline import VideoProcessor, DEVICE

from tactics.frame_sampler import sample_frames
from tactics.gemini_coach import generate_coach_report, TacticsAnalysisError
from tactics.report_writer import render_markdown

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
INPUTS_DIR = os.path.join(BASE_DIR, "inputs")
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

for d in (INPUTS_DIR, UPLOADS_DIR, OUTPUTS_DIR):
    os.makedirs(d, exist_ok=True)

ALLOWED_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

app = Flask(__name__)


# --------------------------------------------------------------------------
# CORS (the frontend is typically opened as a static file / different origin
# from the Flask server, so we allow cross-origin requests explicitly)
# --------------------------------------------------------------------------
@app.after_request
def add_cors_headers(resp):
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return resp


@app.route("/<path:_any>", methods=["OPTIONS"])
@app.route("/", methods=["OPTIONS"])
def cors_preflight(_any=None):
    return ("", 204)


# --------------------------------------------------------------------------
# Lazy model loading (so the server starts instantly and only pays the
# model-loading cost on first use / at startup in the background)
# --------------------------------------------------------------------------
_models_lock = threading.Lock()
_models = {"detector": None, "jersey": None, "ccnn": None, "processor": None, "error": None}


def get_processor():
    with _models_lock:
        if _models["processor"] is not None or _models["error"] is not None:
            if _models["error"] is not None:
                raise RuntimeError(_models["error"])
            return _models["processor"]

        try:
            from ultralytics import YOLO
        except ImportError as e:
            _models["error"] = (
                "ultralytics is not installed. Run: pip install -r requirements.txt"
            )
            raise RuntimeError(_models["error"]) from e

        try:
            detector = YOLO(os.path.join(MODELS_DIR, "detection_best.pt"))
            jersey_model = load_jersey_cnn(os.path.join(MODELS_DIR, "jersey_ocr_best.pt"), device=DEVICE)
            ccnn_model = load_ccnn_filter(os.path.join(MODELS_DIR, "ccnn_best.pt"), device=DEVICE)
        except Exception as e:
            _models["error"] = f"Failed to load models: {e}"
            raise RuntimeError(_models["error"]) from e

        processor = VideoProcessor(detector, jersey_model, ccnn_model, device=DEVICE)
        _models.update(detector=detector, jersey=jersey_model, ccnn=ccnn_model, processor=processor)
        return processor


# --------------------------------------------------------------------------
# In-memory job/progress store (core pipeline)
# --------------------------------------------------------------------------
_jobs_lock = threading.Lock()
_jobs = {}  # filename -> {percent, message, done, error, result, annotated_url, output_path}


def _set_progress(filename, percent=None, message=None, done=None, error=None):
    with _jobs_lock:
        job = _jobs.setdefault(filename, {"percent": 0, "message": "Queued", "done": False, "error": None})
        if percent is not None:
            job["percent"] = percent
        if message is not None:
            job["message"] = message
        if done is not None:
            job["done"] = done
        if error is not None:
            job["error"] = error


def _run_job(filename, input_path):
    output_name = f"annotated_{os.path.splitext(filename)[0]}.mp4"
    output_path = os.path.join(OUTPUTS_DIR, output_name)
    try:
        processor = get_processor()
    except Exception as e:
        _set_progress(filename, done=True, error=str(e))
        return

    def cb(pct, msg):
        _set_progress(filename, percent=pct, message=msg)

    try:
        result = processor.process(input_path, output_path, progress_cb=cb)
        with _jobs_lock:
            _jobs[filename]["result"] = result
            _jobs[filename]["annotated_url"] = f"/outputs/{output_name}"
            _jobs[filename]["output_path"] = output_path
        _set_progress(filename, percent=100, message="Complete", done=True)
    except Exception as e:
        traceback.print_exc()
        _set_progress(filename, done=True, error=f"{type(e).__name__}: {e}")


def _start_job(filename, input_path):
    _set_progress(filename, percent=0, message="Starting...", done=False, error=None)
    with _jobs_lock:
        _jobs[filename]["result"] = None
        _jobs[filename]["annotated_url"] = None
        _jobs[filename]["output_path"] = None
    t = threading.Thread(target=_run_job, args=(filename, input_path), daemon=True)
    t.start()


# --------------------------------------------------------------------------
# Core pipeline routes (unchanged behavior from the original project)
# --------------------------------------------------------------------------
@app.route("/inputs", methods=["GET"])
def list_inputs():
    try:
        files = sorted(
            f for f in os.listdir(INPUTS_DIR)
            if os.path.splitext(f)[1].lower() in ALLOWED_EXT
        )
        return jsonify({"success": True, "files": files})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/upload", methods=["POST"])
def upload_video():
    if "video" not in request.files:
        return jsonify({"success": False, "error": "No 'video' file in request."}), 400
    file = request.files["video"]
    if not file.filename:
        return jsonify({"success": False, "error": "Empty filename."}), 400

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXT:
        return jsonify({"success": False, "error": f"Unsupported file type '{ext}'."}), 400

    safe_name = f"{int(time.time())}_{os.path.basename(file.filename)}"
    save_path = os.path.join(UPLOADS_DIR, safe_name)
    file.save(save_path)

    _start_job(safe_name, save_path)
    return jsonify({"success": True, "filename": safe_name})


@app.route("/process", methods=["POST"])
def process_existing():
    data = request.get_json(silent=True) or {}
    filename = data.get("filename")
    if not filename:
        return jsonify({"success": False, "error": "Missing 'filename'."}), 400

    input_path = os.path.join(INPUTS_DIR, filename)
    if not os.path.isfile(input_path):
        return jsonify({"success": False, "error": f"'{filename}' not found in inputs/."}), 404

    _start_job(filename, input_path)
    return jsonify({"success": True, "filename": filename})


@app.route("/progress/<path:filename>", methods=["GET"])
def progress(filename):
    with _jobs_lock:
        job = _jobs.get(filename)
    if job is None:
        return jsonify({"percent": 0, "message": "No such job.", "done": False, "error": None})
    return jsonify({
        "percent": job["percent"],
        "message": job["message"],
        "done": job["done"],
        "error": job["error"],
    })


@app.route("/result/<path:filename>", methods=["GET"])
def result(filename):
    with _jobs_lock:
        job = _jobs.get(filename)
    if job is None or job.get("result") is None:
        return jsonify({"success": False, "error": "Result not ready yet."}), 404
    return jsonify({
        "success": True,
        "annotated_url": job["annotated_url"],
        "result": job["result"],
    })


@app.route("/outputs/<path:filename>", methods=["GET"])
def serve_output(filename):
    mimetype = "video/mp4" if filename.lower().endswith(".mp4") else None
    return send_from_directory(OUTPUTS_DIR, filename, mimetype=mimetype)


# --------------------------------------------------------------------------
# Tactics routes (new -- generated only when explicitly requested)
# --------------------------------------------------------------------------
_tactics_lock = threading.Lock()
_tactics_jobs = {}  # filename -> {percent, message, done, error, report, markdown}


def _set_tactics_progress(filename, percent=None, message=None, done=None, error=None):
    with _tactics_lock:
        job = _tactics_jobs.setdefault(
            filename, {"percent": 0, "message": "Queued", "done": False, "error": None}
        )
        if percent is not None:
            job["percent"] = percent
        if message is not None:
            job["message"] = message
        if done is not None:
            job["done"] = done
        if error is not None:
            job["error"] = error


def _run_tactics_job(filename, video_path, pipeline_result):
    try:
        _set_tactics_progress(filename, percent=10, message="Sampling frames from annotated video...")
        frames = sample_frames(video_path, max_frames=8)
        if not frames:
            raise RuntimeError("Could not sample any frames from the annotated video.")

        _set_tactics_progress(filename, percent=40, message="Sending match data + frames to Gemini...")
        report = generate_coach_report(pipeline_result, frames)

        # Generate professional tactical visualizations
        _set_tactics_progress(filename, percent=70, message="Generating tactical visualization diagrams...")
        visualizations = {}
        try:
            from tactics.visualization_pipeline import TacticalVisualizationPipeline
            from pathlib import Path
            import re
            
            def _parse_formation_string(f_str: str) -> str:
                if not f_str:
                    return "4-3-3"
                match = re.search(r'\b\d-\d-\d(?:-\d)?\b', f_str)
                if match:
                    found = match.group(0)
                    if found in ["4-3-3", "4-2-3-1", "5-3-2"]:
                        return found
                    if found in ["4-4-2", "3-5-2"]:
                        if "5" in found:
                            return "5-3-2"
                        return "4-3-3"
                if "4-2-3-1" in f_str:
                    return "4-2-3-1"
                if "5-3-2" in f_str:
                    return "5-3-2"
                return "4-3-3"
            
            home_form = _parse_formation_string(report.get("match_overview", {}).get("estimated_formation_team_a", "4-3-3"))
            away_form = _parse_formation_string(report.get("match_overview", {}).get("estimated_formation_team_b", "4-3-3"))
            
            home_data = {
                "formation": home_form,
                "build_up_style": report.get("team_a_analysis", {}).get("build_up_style") or report.get("team_a_analysis", {}).get("playing_style", "Unknown"),
                "pressing_style": report.get("team_a_analysis", {}).get("pressing_style") or report.get("team_a_analysis", {}).get("playing_style", "Unknown"),
                "defensive_shape": report.get("team_a_analysis", {}).get("defensive_shape") or report.get("team_a_analysis", {}).get("playing_style", "Unknown"),
                "attacking_patterns": report.get("team_a_analysis", {}).get("attacking_patterns") or "wing play",
                "transition_style": report.get("team_a_analysis", {}).get("transition_style") or report.get("team_a_analysis", {}).get("transition_play", "Unknown"),
            }
            
            away_data = {
                "formation": away_form,
                "build_up_style": report.get("team_b_analysis", {}).get("build_up_style") or report.get("team_b_analysis", {}).get("playing_style", "Unknown"),
                "pressing_style": report.get("team_b_analysis", {}).get("pressing_style") or report.get("team_b_analysis", {}).get("playing_style", "Unknown"),
                "defensive_shape": report.get("team_b_analysis", {}).get("defensive_shape") or report.get("team_b_analysis", {}).get("playing_style", "Unknown"),
                "attacking_patterns": report.get("team_b_analysis", {}).get("attacking_patterns") or "wing play",
                "transition_style": report.get("team_b_analysis", {}).get("transition_style") or report.get("team_b_analysis", {}).get("transition_play", "Unknown"),
            }
            
            viz_pipeline = TacticalVisualizationPipeline(output_dir=Path(OUTPUTS_DIR))
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            base_name_clean = base_name.replace(' ', '_').replace(':', '')
            
            viz_pipeline.process_single_analysis(home_data, f"{base_name_clean}_team_a", generate_animations=True)
            viz_pipeline.process_single_analysis(away_data, f"{base_name_clean}_team_b", generate_animations=True)
            viz_pipeline.process_match_comparison(home_data, away_data, base_name_clean)
            
            visualizations = {
                "formation_comparison": f"/outputs/{base_name_clean}_formation_comparison.png",
                "tactical_comparison": f"/outputs/{base_name_clean}_tactical_comparison.png",
                "head_to_head": f"/outputs/{base_name_clean}_head_to_head.png",
                "team_a": {
                    "formation": f"/outputs/{base_name_clean}_team_a_formation.png",
                    "tactical_analysis": f"/outputs/{base_name_clean}_team_a_tactical_analysis.png",
                    "defensive_shape": f"/outputs/{base_name_clean}_team_a_defensive_shape.png",
                    "attacking_patterns": f"/outputs/{base_name_clean}_team_a_attacking_patterns.png",
                    "attacking_animation": f"/outputs/{base_name_clean}_team_a_attacking_animation.gif",
                    "defensive_animation": f"/outputs/{base_name_clean}_team_a_defensive_animation.gif"
                },
                "team_b": {
                    "formation": f"/outputs/{base_name_clean}_team_b_formation.png",
                    "tactical_analysis": f"/outputs/{base_name_clean}_team_b_tactical_analysis.png",
                    "defensive_shape": f"/outputs/{base_name_clean}_team_b_defensive_shape.png",
                    "attacking_patterns": f"/outputs/{base_name_clean}_team_b_attacking_patterns.png",
                    "attacking_animation": f"/outputs/{base_name_clean}_team_b_attacking_animation.gif",
                    "defensive_animation": f"/outputs/{base_name_clean}_team_b_defensive_animation.gif"
                }
            }
        except Exception as viz_exc:
            print(f"Error generating tactical visualizations: {viz_exc}")
            traceback.print_exc()

        _set_tactics_progress(filename, percent=85, message="Rendering markdown report...")
        markdown = render_markdown(report, video_label=filename)

        with _tactics_lock:
            job = _tactics_jobs.setdefault(filename, {})
            job["report"] = report
            job["markdown"] = markdown
            job["visualizations"] = visualizations

        # Persist alongside the annotated video for convenience.
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        try:
            import json as _json
            with open(os.path.join(OUTPUTS_DIR, f"{base_name}_coach_report.json"), "w", encoding="utf-8") as f:
                _json.dump(report, f, indent=2)
            with open(os.path.join(OUTPUTS_DIR, f"{base_name}_coach_report.md"), "w", encoding="utf-8") as f:
                f.write(markdown)
            if visualizations:
                with open(os.path.join(OUTPUTS_DIR, f"{base_name}_visualizations.json"), "w", encoding="utf-8") as f:
                    _json.dump(visualizations, f, indent=2)
        except Exception:
            pass  # non-fatal: the report is still returned via the API

        _set_tactics_progress(filename, percent=100, message="Complete", done=True)
    except TacticsAnalysisError as e:
        _set_tactics_progress(filename, done=True, error=str(e))
    except Exception as e:
        traceback.print_exc()
        _set_tactics_progress(filename, done=True, error=f"{type(e).__name__}: {e}")


@app.route("/tactics/<path:filename>", methods=["POST"])
def start_tactics(filename):
    with _jobs_lock:
        job = _jobs.get(filename)

    if job is None or job.get("result") is None or not job.get("output_path"):
        return jsonify({
            "success": False,
            "error": "This video has not finished processing yet. Run detection first.",
        }), 400

    video_path = job["output_path"]
    pipeline_result = job["result"]

    _set_tactics_progress(filename, percent=0, message="Starting tactical analysis...", done=False, error=None)
    with _tactics_lock:
        _tactics_jobs[filename]["report"] = None
        _tactics_jobs[filename]["markdown"] = None

    t = threading.Thread(target=_run_tactics_job, args=(filename, video_path, pipeline_result), daemon=True)
    t.start()
    return jsonify({"success": True, "filename": filename})


@app.route("/tactics/progress/<path:filename>", methods=["GET"])
def tactics_progress(filename):
    with _tactics_lock:
        job = _tactics_jobs.get(filename)
    if job is None:
        return jsonify({"percent": 0, "message": "No such job.", "done": False, "error": None})
    return jsonify({
        "percent": job["percent"],
        "message": job["message"],
        "done": job["done"],
        "error": job["error"],
    })


@app.route("/tactics/result/<path:filename>", methods=["GET"])
def tactics_result(filename):
    with _tactics_lock:
        job = _tactics_jobs.get(filename)
    if job is None or job.get("report") is None:
        return jsonify({"success": False, "error": "Tactical report not ready yet."}), 404
    return jsonify({
        "success": True,
        "report": job["report"],
        "markdown": job["markdown"],
        "visualizations": job.get("visualizations", {})
    })


# --------------------------------------------------------------------------
# Misc / static routes
# --------------------------------------------------------------------------
@app.route("/", methods=["GET"])
def index():
    return send_from_directory(os.path.join(BASE_DIR, "static"), "index.html")


@app.route("/about", methods=["GET"])
def about():
    return send_from_directory(os.path.join(BASE_DIR, "static"), "about.html")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "device": DEVICE})


if __name__ == "__main__":
    print(f"HALO backend starting on http://0.0.0.0:5000  (device={DEVICE})")
    print("Pre-loading models (this can take a while the first time)...")
    try:
        get_processor()
        print("Models loaded OK.")
    except Exception as e:
        print(f"WARNING: model preload failed, will retry on first request: {e}")
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), threaded=True, debug=False)

#!/usr/bin/env python3
"""
Local server for the dashboard.
- Serves static files (HTML, images)
- Proxies Claude API calls (/api/chat)
- Runs pipeline stages 4-6 (/api/pipeline)

Usage:
    python server.py
    # Opens http://localhost:8420/dashboard.html
"""

import os
import sys
import json
import http.server
import urllib.request
import urllib.error
import base64
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

load_dotenv()
API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
PORT = int(os.getenv("PORT", 8420))

# Cache for pipeline state (per session)
_pipeline_state = {}


_last_audio_path = {"path": ""}

def _json_response(handler, code, data):
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.end_headers()
    handler.wfile.write(json.dumps(data).encode())


def _read_body(handler):
    length = int(handler.headers.get("Content-Length", 0))
    return json.loads(handler.rfile.read(length)) if length else {}


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def do_POST(self):
        if self.path == "/api/chat":
            self._handle_chat()
        elif self.path == "/api/pipeline":
            self._handle_pipeline()
        elif self.path == "/api/upload":
            self._handle_upload()
        else:
            self.send_response(404)
            self.end_headers()

    def _handle_upload(self):
        """Receive audio file, save it, run full analysis (librosa + CLAP)."""
        import cgi
        import tempfile

        content_type = self.headers.get('Content-Type', '')

        if 'multipart/form-data' in content_type:
            # Parse multipart form data
            boundary = content_type.split('boundary=')[1].encode()
            length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(length)

            # Extract file data (simple parser)
            parts = body.split(b'--' + boundary)
            audio_bytes = None
            filename = "uploaded_audio.wav"
            for part in parts:
                if b'filename=' in part:
                    # Extract filename
                    header_end = part.find(b'\r\n\r\n')
                    header = part[:header_end].decode('utf-8', errors='ignore')
                    if 'filename="' in header:
                        filename = header.split('filename="')[1].split('"')[0]
                    audio_bytes = part[header_end + 4:].rstrip(b'\r\n--')
                    break

            if not audio_bytes:
                _json_response(self, 400, {"error": "No audio file in upload"})
                return

            # Save to data folder
            save_path = ROOT / "data" / "uploads" / filename
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "wb") as f:
                f.write(audio_bytes)
            print(f"  [UPLOAD] Saved: {save_path}")

            # Run full Python analysis
            try:
                from src.capture.loader import run as load_audio
                from src.spatial_analysis.analyze import run as analyze

                print("  [UPLOAD] Running analysis...")
                audio_data = load_audio(str(save_path))
                analysis = analyze(audio_data, output_dir="outputs/metadata")

                # Run CLAP
                clap_matches = None
                try:
                    from src.audio_embedding.match_environment import run as match_env
                    print("  [UPLOAD] Running CLAP...")
                    clap_matches = match_env(audio_data)
                    if clap_matches:
                        print(f"    CLAP: {clap_matches[0][0]} ({clap_matches[0][1]:.3f})")
                except Exception as e:
                    print(f"    CLAP skipped: {e}")

                # Store for pipeline use
                _pipeline_state["audio_path"] = str(save_path)
                _pipeline_state["analysis"] = analysis
                _pipeline_state["clap_matches"] = clap_matches

                # Format CLAP for response
                clap_response = []
                if clap_matches:
                    for desc, score in clap_matches[:5]:
                        clap_response.append({"description": desc, "similarity": round(float(score), 4)})

                _json_response(self, 200, {
                    "status": "ok",
                    "file": filename,
                    "analysis": analysis,
                    "clap_matches": clap_response,
                })

            except Exception as e:
                import traceback
                traceback.print_exc()
                _json_response(self, 500, {"error": str(e)})
        else:
            _json_response(self, 400, {"error": "Expected multipart/form-data"})

    def _handle_chat(self):
        """Proxy chat to Claude API."""
        body = _read_body(self)

        payload = json.dumps({
            "model": body.get("model", "claude-sonnet-4-20250514"),
            "max_tokens": body.get("max_tokens", 256),
            "messages": body.get("messages", [])
        }).encode()

        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "x-api-key": API_KEY,
                "anthropic-version": "2023-06-01",
            },
            method="POST"
        )

        try:
            with urllib.request.urlopen(req) as resp:
                _json_response(self, 200, json.loads(resp.read()))
        except urllib.error.HTTPError as e:
            error_body = e.read().decode() if e.fp else str(e)
            _json_response(self, e.code, {"error": error_body})

    def _handle_pipeline(self):
        """Run pipeline stages 4→5→6 on the current audio analysis."""
        body = _read_body(self)
        action = body.get("action", "")
        analysis = body.get("analysis", {})

        try:
            if action == "predict":
                # Use server-side analysis if available (from upload)
                if _pipeline_state.get("analysis"):
                    analysis = _pipeline_state["analysis"]
                from src.metadata_predictor.predict import run as predict
                classification = _pipeline_state.get("classification")
                clap_matches = _pipeline_state.get("clap_matches")
                metadata = predict(analysis, classification=classification,
                                   clap_matches=clap_matches, output_dir="outputs/metadata")
                _pipeline_state["metadata"] = metadata
                _json_response(self, 200, {"stage": 4, "result": metadata})

            elif action == "interpret":
                # Stage 5: LLM scene interpretation
                from src.llm_interpreter.interpret import run as interpret
                metadata = body.get("metadata") or _pipeline_state.get("metadata")
                if not metadata:
                    _json_response(self, 400, {"error": "No metadata. Run predict first."})
                    return
                interpretation = interpret(metadata, output_dir="outputs/metadata", use_api=True)
                _pipeline_state["interpretation"] = interpretation
                _json_response(self, 200, {"stage": 5, "result": interpretation})

            elif action == "generate":
                # Stage 6: Image generation
                from src.image_generation.generate import run as generate
                interpretation = body.get("interpretation") or _pipeline_state.get("interpretation")
                if not interpretation:
                    _json_response(self, 400, {"error": "No interpretation. Run interpret first."})
                    return
                img_method = body.get("method", "auto")
                # Single image mode — only generate front view
                if body.get("single"):
                    front_prompt = interpretation["prompts"].get("front", "")
                    interpretation = dict(interpretation)
                    interpretation["prompts"] = {"front": front_prompt}
                gen_result = generate(interpretation, output_dir="outputs/images", method=img_method)
                # Encode generated images as base64 for dashboard display
                images_b64 = {}
                for view, path in gen_result.get("image_paths", {}).items():
                    if Path(path).exists():
                        with open(path, "rb") as f:
                            images_b64[view] = base64.b64encode(f.read()).decode()
                gen_result["images_b64"] = images_b64
                _pipeline_state["generation"] = gen_result
                _json_response(self, 200, {"stage": 6, "result": gen_result})

            elif action == "full":
                # Run stages 4→5→6 sequentially
                # Create a timestamped run folder
                from datetime import datetime
                from src.metadata_predictor.predict import run as predict
                from src.llm_interpreter.interpret import run as interpret
                from src.image_generation.generate import run as generate

                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                run_dir = ROOT / "outputs" / "runs" / ts
                run_dir.mkdir(parents=True, exist_ok=True)

                print(f"\n  [PIPELINE] Run: {ts}")

                # Use server-side analysis if available (from upload)
                if _pipeline_state.get("analysis"):
                    analysis = _pipeline_state["analysis"]
                    print("  [PIPELINE] Using uploaded audio analysis")

                # Use cached CLAP if available
                if _pipeline_state.get("clap_matches"):
                    clap_matches = _pipeline_state["clap_matches"]
                    if clap_matches:
                        print(f"  [PIPELINE] Using cached CLAP: {clap_matches[0][0]} ({clap_matches[0][1]:.3f})")

                # Run YAMNet classification if available
                classification = None
                try:
                    from src.spatial_analysis.classify import run as classify
                    print("  [PIPELINE] Running YAMNet classification...")
                    # Find actual audio file path
                    audio_path = analysis.get("file", "")
                    # Try resolving relative to project root
                    if not Path(audio_path).exists():
                        # Search data folder for matching file (case-insensitive)
                        ap_lower = audio_path.lower()
                        for wav in (ROOT / "data").rglob("*.wav"):
                            if wav.name.lower() in ap_lower or ap_lower in wav.name.lower() or ap_lower.replace(' ','') in wav.name.lower().replace(' ',''):
                                audio_path = str(wav)
                                print(f"    Found audio: {audio_path}")
                                break
                    if audio_path and Path(audio_path).exists():
                        from src.capture.loader import run as load_audio
                        audio_data = load_audio(audio_path)
                        classification = classify(audio_data)
                        if classification.get("top_classes"):
                            top3 = [f"{c['class_name']} ({c['score']:.2f})" for c in classification["top_classes"][:3]]
                            print(f"    YAMNet: {', '.join(top3)}")
                except Exception as e:
                    print(f"    YAMNet skipped: {e}")
                else:
                    if not audio_path or not Path(audio_path).exists():
                        print(f"    YAMNet skipped: audio file not found ({audio_path})")

                # Run CLAP environment matching if available
                clap_matches = None
                try:
                    from src.audio_embedding.match_environment import run as match_env
                    if audio_path and Path(audio_path).exists():
                        print("  [PIPELINE] Running CLAP environment matching...")
                        if not locals().get('audio_data'):
                            from src.capture.loader import run as load_audio
                            audio_data = load_audio(audio_path)
                        clap_matches = match_env(audio_data)
                        if clap_matches:
                            print(f"    CLAP: {clap_matches[0][0]} ({clap_matches[0][1]:.3f})")
                except Exception as e:
                    print(f"    CLAP skipped: {e}")
                else:
                    if not audio_path or not Path(audio_path).exists():
                        print(f"    CLAP skipped: audio file not found ({audio_path})")

                print("  [PIPELINE] Stage 4: Metadata prediction...")
                metadata = predict(analysis, classification=classification,
                                   clap_matches=clap_matches, output_dir=str(run_dir))

                print("  [PIPELINE] Stage 5: LLM interpretation...")
                interpretation = interpret(metadata, output_dir=str(run_dir), use_api=True)

                # Save full LLM text to run folder
                llm_txt_path = run_dir / "scene_description.txt"
                with open(llm_txt_path, "w") as f:
                    f.write(f"SCENE INTERPRETATION\n")
                    f.write(f"{'='*50}\n")
                    f.write(f"Method: {interpretation.get('interpretation_method','')}\n")
                    f.write(f"Scale: {interpretation.get('spatial_scale','')}\n")
                    f.write(f"Lighting: {interpretation.get('lighting_condition','')}\n")
                    f.write(f"Materials: {', '.join(interpretation.get('material_palette',[]))}\n\n")
                    f.write(f"DESCRIPTION:\n{interpretation.get('scene_paragraph','')}\n\n")
                    f.write(f"PROMPTS:\n")
                    for view, prompt in interpretation.get("prompts", {}).items():
                        f.write(f"  [{view.upper()}] {prompt}\n")
                print(f"  Saved: {llm_txt_path}")

                img_method = body.get("method", "auto")
                if body.get("single"):
                    interpretation = dict(interpretation)
                    interpretation["prompts"] = {"front": interpretation["prompts"].get("front", "")}
                print(f"  [PIPELINE] Stage 6: Image generation ({img_method})...")
                gen_result = generate(interpretation, output_dir=str(run_dir), method=img_method)

                # Encode images
                images_b64 = {}
                for view, path in gen_result.get("image_paths", {}).items():
                    if Path(path).exists():
                        with open(path, "rb") as f:
                            images_b64[view] = base64.b64encode(f.read()).decode()
                gen_result["images_b64"] = images_b64

                _pipeline_state.update({
                    "metadata": metadata,
                    "interpretation": interpretation,
                    "generation": gen_result,
                })

                _json_response(self, 200, {
                    "stages": [4, 5, 6],
                    "metadata": metadata,
                    "interpretation": interpretation,
                    "generation": gen_result,
                    "run_dir": f"outputs/runs/{ts}",
                })

            elif action == "generate_custom":
                # Direct image generation with custom prompt
                from src.image_generation.generate import run as generate
                custom_prompt = body.get("prompt", "")
                img_method = body.get("method", "gpt-image")
                if not custom_prompt:
                    _json_response(self, 400, {"error": "No prompt provided."})
                    return

                suffix = "shot on 35mm lens, f/2.8, hyper-realistic, raw photo, unedited, Kodak Portra 400, cinematic lighting, 8k, highly detailed, physically based rendering"
                full_prompt = f"{custom_prompt}, {suffix}"

                interp = {
                    "prompts": {"front": full_prompt},
                    "source_metadata": {"environment_type": "custom"},
                    "interpretation_method": "custom",
                    "spatial_scale": "custom",
                    "lighting_condition": "custom",
                    "material_palette": [],
                }

                from datetime import datetime
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                run_dir = ROOT / "outputs" / "runs" / ts
                run_dir.mkdir(parents=True, exist_ok=True)

                # Save prompt
                with open(run_dir / "custom_prompt.txt", "w") as f:
                    f.write(full_prompt)

                gen_result = generate(interp, output_dir=str(run_dir), method=img_method)
                images_b64 = {}
                for view, path in gen_result.get("image_paths", {}).items():
                    if Path(path).exists():
                        with open(path, "rb") as f:
                            images_b64[view] = base64.b64encode(f.read()).decode()
                gen_result["images_b64"] = images_b64

                _json_response(self, 200, {
                    "stage": 6,
                    "result": gen_result,
                    "run_dir": f"outputs/runs/{ts}",
                    "prompt_used": full_prompt,
                })

            else:
                _json_response(self, 400, {"error": f"Unknown action: {action}"})

        except Exception as e:
            import traceback
            traceback.print_exc()
            _json_response(self, 500, {"error": str(e)})

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def log_message(self, fmt, *args):
        msg = fmt % args
        if "GET" in msg and any(ext in msg for ext in [".js", ".css", ".woff", ".png", ".ico"]):
            return
        print(f"  {msg}")


if __name__ == "__main__":
    if not API_KEY:
        print("  WARNING: No ANTHROPIC_API_KEY in .env — LLM interpretation will use templates")

    # Check for embedded dashboard
    embedded = ROOT / "outputs" / "metadata"
    embedded_files = sorted(embedded.glob("*_dashboard.html")) if embedded.exists() else []
    if embedded_files:
        serve_path = f"/outputs/metadata/{embedded_files[-1].name}"
    else:
        serve_path = "/dashboard.html"

    print(f"\n  ╔══════════════════════════════════════╗")
    print(f"  ║  Spatial Audio Analysis Dashboard    ║")
    print(f"  ║  http://localhost:{PORT}{serve_path}  ║")
    print(f"  ╚══════════════════════════════════════╝\n")

    server = http.server.HTTPServer(("", PORT), Handler)
    import webbrowser
    webbrowser.open(f"http://localhost:{PORT}{serve_path}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Stopped.")

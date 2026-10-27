"""Traffic sign action lookup API and phone page.

Reads sign_actions.csv and returns the recorded action for each requested sign.
The phone page at / lets you open a dashcam MP4. The server reads that clip at
24 frames per second and classifies signs with the trained VGG16 model, then
the page asks /predict for the action text.

The video check uses the libraries already installed for the model:

    ../traffic-sign-alert/.venv_linux/bin/pip install -r requirements.txt
    ../traffic-sign-alert/.venv_linux/bin/uvicorn main:app --host 0.0.0.0 --port 8000

Open http://<this-computer-lan-ip>:8000 on the phone, then choose an MP4.
"""

from __future__ import annotations

import asyncio
import csv
import re
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "sign_actions.csv"
STATIC_DIR = BASE_DIR / "static"
SAMPLE_CLIP = BASE_DIR / "samples" / "sample_clip.mp4"
DASHCAM_CLIP = BASE_DIR / "camera00_test_5_6s.mp4"
LONG_CLIP = BASE_DIR / "camera00_second_clip_197_frames.mp4"
SIGN_COLUMN = "sign"
ACTION_COLUMN = "action"
MAX_UPLOAD_BYTES = 80 * 1024 * 1024
VIDEO_SUFFIXES = {".mp4", ".m4v", ".mov"}

# Spaces and hyphens are folded into underscores so "No Entry" matches no_entry.
_SEPARATORS = re.compile(r"[\s\-]+")
_REPEATED_UNDERSCORES = re.compile(r"_+")


def normalize_sign(name: str) -> str:
    """Fold case and separators without changing the rest of the sign id."""
    folded = _SEPARATORS.sub("_", name.strip().lower())
    return _REPEATED_UNDERSCORES.sub("_", folded).strip("_")


def load_sign_actions(path: Path = CSV_PATH) -> dict[str, str]:
    """Load the sign -> action table. Keys are normalized sign ids."""
    if not path.is_file():
        raise FileNotFoundError(f"Sign action file not found: {path}")

    mapping: dict[str, str] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header row")

        fields = [name.strip() for name in reader.fieldnames]
        missing = [column for column in (SIGN_COLUMN, ACTION_COLUMN) if column not in fields]
        if missing:
            raise ValueError(
                f"{path} is missing column(s) {missing}; found {fields}"
            )

        for line_number, row in enumerate(reader, start=2):
            sign = (row.get(SIGN_COLUMN) or "").strip()
            action = (row.get(ACTION_COLUMN) or "").strip()
            if not sign:
                raise ValueError(f"{path} line {line_number} has an empty sign name")
            key = normalize_sign(sign)
            if key in mapping:
                raise ValueError(f"Duplicate sign '{sign}' in {path} at line {line_number}")
            mapping[key] = action

    if not mapping:
        raise ValueError(f"{path} contains no sign rows")
    return mapping


class PredictRequest(BaseModel):
    signs: list[str] = Field(
        ...,
        description="Sign names in the order they should be returned.",
    )


class SignAction(BaseModel):
    sign: str
    action: str | None = None


app = FastAPI(
    title="Traffic Sign Alert API",
    description=(
        "Looks up the driver action recorded in sign_actions.csv for each sign name."
    ),
    version="1.0.0",
)

try:
    SIGN_ACTIONS = load_sign_actions()
except (OSError, ValueError, csv.Error) as exc:
    raise RuntimeError(f"Failed to load {CSV_PATH.name}: {exc}") from exc


@app.get("/health")
def health() -> dict[str, int | str | bool]:
    model_error = _model_error()
    return {
        "signs_loaded": len(SIGN_ACTIONS),
        "model_ready": model_error is None,
        "model_detail": model_error or "ready",
    }


@app.get("/")
def home() -> FileResponse:
    return FileResponse(
        STATIC_DIR / "index.html",
        headers={"Cache-Control": "no-store"},
    )


@app.get("/sample-clip.mp4")
def sample_clip() -> FileResponse:
    if not SAMPLE_CLIP.is_file():
        raise HTTPException(status_code=404, detail="Sample clip is not on this machine.")
    return FileResponse(SAMPLE_CLIP, media_type="video/mp4", filename="sample_clip.mp4")


@app.get("/sw.js")
def service_worker() -> FileResponse:
    return FileResponse(
        STATIC_DIR / "sw.js",
        media_type="application/javascript",
        headers={"Cache-Control": "no-cache"},
    )


@app.get("/long-clip.mp4")
def long_clip() -> FileResponse:
    if not LONG_CLIP.is_file():
        raise HTTPException(status_code=404, detail="camera00_second_clip_197_frames.mp4 was not found.")
    return FileResponse(
        LONG_CLIP,
        media_type="video/mp4",
        filename="camera00_second_clip_197_frames.mp4",
    )


@app.get("/dashcam.mp4")
def dashcam_clip() -> FileResponse:
    if not DASHCAM_CLIP.is_file():
        raise HTTPException(status_code=404, detail="camera00_test_5_6s.mp4 was not found.")
    return FileResponse(
        DASHCAM_CLIP,
        media_type="video/mp4",
        filename="camera00_test_5_6s.mp4",
    )


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)) -> dict:
    """Find sign ids in an uploaded dashcam clip.

    The phone page sends those ids to /predict for the CSV action text.
    """
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in VIDEO_SUFFIXES:
        raise HTTPException(status_code=400, detail="Choose an MP4 dashcam clip.")
    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="That video file is empty.")
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Video is larger than 80 MB.")

    model_error = _model_error()
    if model_error:
        raise HTTPException(status_code=503, detail=model_error)

    import video_analysis

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
        handle.write(payload)
        clip_path = Path(handle.name)
    try:
        return await asyncio.to_thread(
            video_analysis.analyze_video,
            clip_path,
            list(SIGN_ACTIONS),
        )
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        clip_path.unlink(missing_ok=True)


def _model_error() -> str | None:
    try:
        import video_analysis
    except ImportError:
        return (
            "Video checking needs torch, torchvision, and opencv. "
            "Start this server with traffic-sign-alert/.venv_linux."
        )
    return video_analysis.model_is_available()


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.post("/predict", response_model=list[SignAction])
def predict(body: PredictRequest) -> list[SignAction]:
    """Return CSV actions in the same order as the input signs.

    A sign that does not match a row has action set to null. No action text
    is generated for names that are absent from the CSV.
    """
    results: list[SignAction] = []
    for sign in body.signs:
        key = normalize_sign(sign)
        action = SIGN_ACTIONS.get(key) if key else None
        results.append(SignAction(sign=sign, action=action))
    return results

# Traffic Sign Alert API

REST API that looks up a driver action for each traffic sign name in `sign_actions.csv`.

The CSV has two columns:

- `sign` — sign id, such as `no_entry` or `speed_limit_50kmph`
- `action` — the action text stored for that sign

`POST /predict` returns those action strings. It does not generate actions for names that are not in the file.

## Setup

From this directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
uvicorn main:app --host 127.0.0.1 --port 8000
```

Interactive docs are at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

## Phone page

Open `/` after starting the server with the environment that already has the sign model (PyTorch, torchvision, and OpenCV):

```bash
../traffic-sign-alert/.venv_linux/bin/pip install -r requirements.txt
../traffic-sign-alert/.venv_linux/bin/uvicorn main:app --host 0.0.0.0 --port 8000
```

On the phone, open `http://<this-computer-lan-ip>:8000`. Choose a dashcam MP4, or use the sample clip. The server reads up to 10 seconds at 24 frames per second, classifies sign regions with `traffic-sign-alert/models/vgg16_signs_general.pt`, and the page sends the resulting sign ids to `POST /predict`.

There is no dashcam recording in this project yet. The sample clip is a sign photo placed on a plain background so the page can be tried before you add your own MP4. A real dashcam file belongs at `traffic-sign-alert/data/raw/sample_dashcam.mp4` if you also want the older desktop demo.

## Predict

```bash
curl -s -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"signs": ["no_entry", "Stop", "speed_limit_50kmph"]}'
```

Request body:

```json
{"signs": ["Stop", "No Entry", "Speed Limit"]}
```

Response: a JSON array, one object per input sign, in the same order. Each object has `sign` (the name you sent) and `action` (the CSV text, or `null` when that name is not a row in the file).

Matching is case-insensitive, and spaces or hyphens are treated as underscores, so `No Entry` matches `no_entry`. A name must still match one full sign id. `Speed Limit` does not match `speed_limit_50kmph`, and `Stop` is not in the CSV, so both come back with `"action": null`.

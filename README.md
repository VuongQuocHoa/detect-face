# InsightFace Recognition API

<div align="center">

[![Live Demo on Render](https://img.shields.io/badge/🌐_Live_Demo-Render-00f0ff?style=for-the-badge&logo=render&logoColor=white)](https://vuongquochoa-detect-face.onrender.com/)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-181717?style=for-the-badge&logo=github)](https://github.com/VuongQuocHoa/detect-face)

</div>

Production-oriented FastAPI service for registering and recognizing faces with
InsightFace and persistent ChromaDB storage.

## Setup

From `D:\NHAP\dectect_face`:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The InsightFace `buffalov` model is downloaded by InsightFace on first use.
Embeddings are stored in `D:\NHAP\dectect_face\data\chroma`.

## Endpoints

- `GET /health`
- `POST /api/v1/register` with multipart fields `user_id`, `name`, and `file`
- `POST /api/v1/recognize` with multipart field `file`

Registration uses the largest detected face. Recognition returns all detected
faces, their bounding boxes, identity, and cosine distance. Unknown faces have
`id` and `distance` set to `null`.

## Webcam client

With the API running, start the laptop webcam client:

```powershell
python test_cam.py
```

Press `q` or `Esc` to close the video window. Use `--camera 1` when the
webcam is not camera index 0, or `--interval 0.5` to reduce API requests.

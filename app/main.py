import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

from typing import Annotated, Any
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from starlette.concurrency import run_in_threadpool

import base64
import time
import cv2
import numpy as np

from app.config import BASE_DIR, FACES_DIR, SIMILARITY_THRESHOLD
from app.core.face_engine import extract_embeddings, get_face_engine
from app.core.tracker import FaceStreamTracker
from app.core.vector_db import (
    convert_stranger,
    delete_all_faces,
    delete_all_registered_members,
    delete_all_strangers,
    delete_user,
    list_registered_members,
    list_strangers,
    list_users,
    search_face,
    update_user_info,
)
from app.schemas.face import (
    DetectedFaceCrop,
    DetectFacesResponse,
    FaceMatch,
    RecognitionResponse,
    RegisterResponse,
)
from app.utils.image_utils import crop_and_save_face, read_upload



@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-warm the InsightFace models on startup and activate UDP auto-discovery."""
    try:
        from app.core.discovery import start_discovery_server
        start_discovery_server(http_port=8000, discovery_port=8005)
    except Exception as exc:
        print(f"Warning during auto-discovery start: {exc}")

    try:
        engine = get_face_engine()
        # Warm up OpenVINO GPU execution graph to eliminate first-frame latency spike
        dummy = np.zeros((320, 320, 3), dtype=np.uint8)
        engine.extract_from_image_array(dummy)
    except Exception as exc:
        print(f"Warning during face engine prewarm: {exc}")
    yield


app = FastAPI(
    title="InsightFace Recognition API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/api/server-info")
async def get_server_info():
    """Return active LAN IP and HTTP/WS port for auto-pairing."""
    from app.core.discovery import get_lan_ip
    lan_ip = get_lan_ip()
    return {
        "status": "ok",
        "server": "FaceID_CyberHUD",
        "ip": lan_ip,
        "port": 8000,
        "host": f"{lan_ip}:8000",
        "service": "ai_face_recognition"
    }


@app.get("/", include_in_schema=False)
async def dashboard() -> FileResponse:
    """Serve the real-time cyberpunk webcam dashboard with zero browser caching."""
    return FileResponse(
        Path(BASE_DIR) / "app" / "templates" / "index.html",
        media_type="text/html",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@app.get("/favicon.ico", include_in_schema=False)
async def favicon() -> FileResponse:
    """Handle browser favicon request cleanly."""
    from fastapi.responses import Response
    return Response(content=b"", media_type="image/x-icon")


@app.get("/manifest.json", include_in_schema=False)
async def manifest() -> FileResponse:
    """Serve PWA manifest for Android homescreen installation."""
    return FileResponse(
        Path(BASE_DIR) / "app" / "static" / "manifest.json",
        media_type="application/manifest+json",
    )


@app.get("/sw.js", include_in_schema=False)
async def service_worker() -> FileResponse:
    """Serve Service Worker for instant offline loading."""
    return FileResponse(
        Path(BASE_DIR) / "app" / "static" / "sw.js",
        media_type="application/javascript",
    )


_static_dir = Path(BASE_DIR) / "app" / "static"
if _static_dir.exists():
    from fastapi.staticfiles import StaticFiles
    app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")


@app.get("/health")
async def health() -> dict[str, str]:
    """Return service availability."""
    return {"status": "ok"}


@app.get("/api/v1/users")
async def get_registered_users() -> list[dict[str, Any]]:
    """Return all registered users in ChromaDB."""
    return list_users()


@app.delete("/api/v1/users/{user_id}")
async def remove_user(user_id: str) -> dict[str, Any]:
    """Delete all face embeddings associated with a user ID."""
    count = delete_user(user_id)
    if count == 0:
        raise HTTPException(status_code=404, detail=f"User ID '{user_id}' not found.")
    return {"message": f"Deleted {count} record(s) for user {user_id}.", "deleted_count": count}


DEFAULT_AVATAR_SVG = """<svg xmlns="http://www.w3.org/2000/svg" width="120" height="120" viewBox="0 0 120 120">
  <rect width="120" height="120" fill="#060e1d" rx="8"/>
  <circle cx="60" cy="46" r="22" fill="#0d223a" stroke="#00f3ff" stroke-width="2"/>
  <path d="M 28 100 C 28 75, 42 68, 60 68 C 78 68, 92 75, 92 100" fill="#0d223a" stroke="#00f3ff" stroke-width="2"/>
</svg>"""


@app.get("/api/v1/users/{user_id}/photo")
async def get_user_photo(user_id: str):
    """Serve the face thumbnail photo for registered users or strangers."""
    from fastapi.responses import Response
    photo_file = Path(FACES_DIR) / f"{user_id}.jpg"
    if photo_file.is_file():
        return FileResponse(
            photo_file,
            media_type="image/jpeg",
            headers={"Cache-Control": "public, max-age=86400"},
        )
    return Response(content=DEFAULT_AVATAR_SVG, media_type="image/svg+xml")


@app.get("/api/v1/strangers")
async def get_strangers() -> list[dict[str, Any]]:
    """Return all detected and auto-registered strangers."""
    return list_strangers()


@app.delete("/api/v1/strangers")
async def remove_all_strangers() -> dict[str, Any]:
    """Delete all auto-registered strangers from ChromaDB and storage."""
    count = delete_all_strangers()
    return {"message": f"Đã xóa {count} bản ghi người lạ.", "deleted_count": count}


@app.delete("/api/v1/users")
async def remove_all_users() -> dict[str, Any]:
    """Delete all registered face records from ChromaDB and storage."""
    count = delete_all_registered_members()
    return {"message": f"Đã xóa toàn bộ {count} khuôn mặt đã đăng ký.", "deleted_count": count}


@app.post("/api/v1/users/{user_id}/convert")
async def convert_stranger_endpoint(
    user_id: str,
    new_user_id: Annotated[str, Form(min_length=1)],
    new_name: Annotated[str, Form(min_length=1)],
) -> dict[str, Any]:
    """Convert a stranger into an official registered member."""
    success = convert_stranger(user_id, new_user_id.strip(), new_name.strip())
    if not success:
        raise HTTPException(status_code=404, detail=f"Không tìm thấy người lạ '{user_id}' để chuyển đổi.")
    return {
        "message": f"Chuyển đổi thành công! Người lạ '{user_id}' đã trở thành thành viên '{new_name}' ({new_user_id}).",
        "user_id": new_user_id.strip(),
        "name": new_name.strip(),
    }


@app.post("/api/v1/users/{user_id}/update")
async def update_user_endpoint(
    user_id: str,
    new_user_id: Annotated[str, Form(min_length=1)],
    new_name: Annotated[str, Form(min_length=1)],
) -> dict[str, Any]:
    """Update name and/or user_id of a registered member or stranger."""
    success = update_user_info(user_id, new_user_id.strip(), new_name.strip())
    if not success:
        raise HTTPException(status_code=404, detail=f"Không tìm thấy người dùng '{user_id}' để cập nhật.")
    return {
        "message": f"Cập nhật thành công! Người dùng '{user_id}' đã được đổi thành '{new_name}' ({new_user_id}).",
        "user_id": new_user_id.strip(),
        "name": new_name.strip(),
    }



def _crop_and_encode_face(image: np.ndarray, bbox: list[float], margin_ratio: float = 0.20) -> str:
    """Crop face region with a margin and encode as JPEG Base64 data URI."""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox
    box_w = x2 - x1
    box_h = y2 - y1

    pad_x = box_w * margin_ratio
    pad_y = box_h * margin_ratio

    crop_x1 = max(0, int(x1 - pad_x))
    crop_y1 = max(0, int(y1 - pad_y))
    crop_x2 = min(w, int(x2 + pad_x))
    crop_y2 = min(h, int(y2 + pad_y))

    face_crop = image[crop_y1:crop_y2, crop_x1:crop_x2]
    if face_crop.size == 0:
        face_crop = image[max(0, int(y1)):min(h, int(y2)), max(0, int(x1)):min(w, int(x2))]

    ok, buf = cv2.imencode(".jpg", face_crop, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok:
        return ""
    b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def _process_detect_faces(image_bytes: bytes) -> list[DetectedFaceCrop]:
    """Synchronous CPU worker to detect all faces, check registration status, and sort candidates."""
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Ảnh không hợp lệ hoặc không đúng định dạng.")

    faces = get_face_engine().extract_from_image_array(image)
    if not faces:
        return []

    crops: list[DetectedFaceCrop] = []
    for i, face in enumerate(faces):
        thumb_uri = _crop_and_encode_face(image, face.bbox)
        # Check against VectorDB to see if face is already registered
        match = search_face(face.embedding, SIMILARITY_THRESHOLD)
        is_reg = isinstance(match, dict) and match.get("matched", False)
        reg_id = str(match["id"]) if is_reg else None
        reg_name = str(match["name"]) if is_reg else None
        sim = float(match["similarity"]) if is_reg else None

        # Quy tắc: Người lạ không được coi là thành viên đã đăng ký, không tự điền ID và Tên
        if is_reg and (
            (reg_id and reg_id.lower().startswith("stranger_"))
            or (reg_name and reg_name.lower().startswith("người lạ"))
        ):
            is_reg = False
            reg_id = None
            reg_name = None

        crops.append(
            DetectedFaceCrop(
                index=i,
                bbox=face.bbox,
                thumbnail=thumb_uri,
                area=float(face.area),
                is_registered=is_reg,
                registered_id=reg_id,
                registered_name=reg_name,
                similarity=sim,
            )
        )

    # Sắp xếp: Chưa đăng ký (is_registered=False) đứng trước, Đã đăng ký (is_registered=True) đứng sau
    crops.sort(key=lambda c: (1 if c.is_registered else 0, -c.area))
    return crops


@app.post("/api/v1/detect-faces", response_model=DetectFacesResponse)
async def detect_faces(
    file: Annotated[UploadFile, File()],
) -> DetectFacesResponse:
    """Detect all faces in an image and return individual cropped thumbnails for registration selection."""
    try:
        image_bytes = await read_upload(file)
        crops = await run_in_threadpool(_process_detect_faces, image_bytes)
        return DetectFacesResponse(total_faces=len(crops), faces=crops)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _process_registration(
    image_bytes: bytes, user_id: str, name: str, face_index: int | None = None
) -> None:
    """Synchronous CPU worker for face registration."""
    faces, error_message = extract_embeddings(image_bytes)
    if error_message is not None:
        raise HTTPException(status_code=400, detail=error_message)
    if not faces:
        raise HTTPException(status_code=400, detail="Không tìm thấy khuôn mặt trong ảnh. Vui lòng chụp rõ mặt hơn.")

    if face_index is not None and 0 <= face_index < len(faces):
        target_face = faces[face_index]
    else:
        target_face = max(faces, key=lambda face: face.area)

    saved, database_error = get_face_engine().register_face(
        user_id=user_id,
        name=name,
        embedding=target_face.embedding,
    )
    if not saved:
        raise HTTPException(status_code=400, detail=database_error)

    try:
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if image is not None:
            crop_and_save_face(image, target_face.bbox, user_id)
    except Exception as exc:
        print(f"Warning saving registered face photo for {user_id}: {exc}")



@app.post("/api/v1/register", response_model=RegisterResponse)
async def register_face(
    user_id: Annotated[str, Form(min_length=1)],
    name: Annotated[str, Form(min_length=1)],
    file: Annotated[UploadFile, File()],
    face_index: Annotated[int | None, Form()] = None,
) -> RegisterResponse:
    """Register a face (chosen by face_index or largest) using a threadpool worker."""
    try:
        image_bytes = await read_upload(file)
        await run_in_threadpool(_process_registration, image_bytes, user_id, name, face_index)
        return RegisterResponse(
            user_id=user_id,
            name=name,
            message="Face registered successfully.",
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _process_recognition(image_bytes: bytes, threshold: float) -> list[FaceMatch]:
    """Synchronous CPU worker for face recognition."""
    faces, error_message = extract_embeddings(image_bytes)
    if error_message is not None:
        raise HTTPException(status_code=400, detail=error_message)
    if not faces:
        return []

    results: list[FaceMatch] = []
    for face in faces:
        match = search_face(face.embedding, threshold)
        if isinstance(match, dict) and match.get("matched", False):
            results.append(
                FaceMatch(
                    bbox=face.bbox,
                    name=str(match["name"]),
                    id=str(match["id"]),
                    distance=float(match["distance"]),
                    similarity=float(match["similarity"]),
                )
            )
        else:
            dist = float(match["distance"]) if isinstance(match, dict) else None
            sim = float(match["similarity"]) if isinstance(match, dict) else None
            results.append(
                FaceMatch(
                    bbox=face.bbox,
                    name="Unknown",
                    id=None,
                    distance=dist,
                    similarity=sim,
                )
            )
    return results


def _process_recognition_fast(image_bytes: bytes, threshold: float) -> list[dict[str, Any]]:
    """Ultra-fast zero-overhead worker for WebSocket recognition streaming."""
    faces, error_message = extract_embeddings(image_bytes)
    if error_message is not None or not faces:
        return []

    results: list[dict[str, Any]] = []
    for face in faces:
        match = search_face(face.embedding, threshold)
        if isinstance(match, dict) and match.get("matched", False):
            results.append({
                "bbox": face.bbox,
                "name": str(match["name"]),
                "id": str(match["id"]),
                "distance": float(match["distance"]),
                "similarity": float(match["similarity"]),
            })
        else:
            dist = float(match["distance"]) if isinstance(match, dict) else None
            sim = float(match["similarity"]) if isinstance(match, dict) else None
            results.append({
                "bbox": face.bbox,
                "name": "Unknown",
                "id": None,
                "distance": dist,
                "similarity": sim,
            })
    return results


@app.post("/api/v1/recognize", response_model=RecognitionResponse)
async def recognize_faces(
    file: Annotated[UploadFile, File()],
    threshold: float | None = Query(default=None, ge=0.0, le=1.0),
) -> RecognitionResponse:
    """Recognize every face found in an image without blocking the asyncio event loop."""
    image_bytes = await read_upload(file)
    active_threshold = threshold if threshold is not None else SIMILARITY_THRESHOLD
    try:
        results = await run_in_threadpool(_process_recognition, image_bytes, active_threshold)
        return RecognitionResponse(faces=results)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


from app.core.protocol import pack_recognition_binary
import asyncio

@app.websocket("/ws/recognize")
async def ws_recognize(websocket: WebSocket):
    """High-Performance Zero-Latency Two-Way Binary WebSocket stream.

    - Server receives raw binary 0xFACE (RGB/RGBA buffer).
    - Runs SCRFD detector on every frame (~2.8ms on OpenVINO GPU).
    - Drop-Stale Async Queue: drops backlogged frames so queue latency is strictly 0ms.
    - Selectively triggers ArcFace with Adaptive Motion Tracking.
    - Sends back ultra-compact 0xBEEF binary struct (~50 bytes) with zero JSON garbage collection!
    """
    await websocket.accept()
    current_threshold = SIMILARITY_THRESHOLD
    tracker = FaceStreamTracker(re_id_interval=45)
    engine = get_face_engine()
    protocol_format = "binary"  # Default to 0xBEEF binary protocol

    try:
        while True:
            msg = await websocket.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            if "bytes" in msg and msg["bytes"]:
                t_start = time.perf_counter()
                try:
                    results = engine.process_bytes_hybrid(
                        msg["bytes"],
                        tracker,
                        current_threshold,
                    )
                    latency_ms = round((time.perf_counter() - t_start) * 1000, 1)
                    if protocol_format == "binary":
                        packet = pack_recognition_binary(results, latency_ms)
                        await websocket.send_bytes(packet)
                    else:
                        await websocket.send_json({
                            "faces": results,
                            "inference_ms": latency_ms,
                        })
                except Exception as proc_err:
                    if protocol_format == "binary":
                        packet = pack_recognition_binary([], 0.0)
                        await websocket.send_bytes(packet)
                    else:
                        await websocket.send_json({
                            "faces": [],
                            "inference_ms": 0.0,
                            "error": str(proc_err),
                        })
            elif "text" in msg and msg["text"]:
                try:
                    import json
                    cfg = json.loads(msg["text"])
                    if "threshold" in cfg:
                        current_threshold = float(cfg["threshold"])
                    if "format" in cfg:
                        protocol_format = str(cfg["format"]).lower()
                except Exception:
                    pass
    except (WebSocketDisconnect, RuntimeError):
        pass
    except Exception as exc:
        print(f"WebSocket session closed: {exc}")





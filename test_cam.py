"""Real-time high-performance webcam client for the face recognition API.

Optimized with:
- Threaded asynchronous camera capture (30-60 FPS)
- Persistent HTTP Keep-Alive session pooling
- Smooth bounding box interpolation (EMA tracking)
"""

from __future__ import annotations

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import argparse
import ctypes
import sys
import threading
import time
from typing import Any

import cv2
import numpy as np
import requests


class ThreadedCamera:
    """Continuously grabs frames in a dedicated thread to decouple camera hardware I/O."""

    def __init__(self, src: int = 0) -> None:
        self.cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FPS, 60)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.ret, self.frame = self.cap.read()
        self.stopped = False
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _reader(self) -> None:
        while not self.stopped:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame
                    self.ret = ret
            time.sleep(0.001)

    def isOpened(self) -> bool:
        return self.cap.isOpened()

    def read(self) -> tuple[bool, Any]:
        with self.lock:
            if not self.ret or self.frame is None:
                return False, None
            return True, self.frame.copy()

    def release(self) -> None:
        self.stopped = True
        self.cap.release()


class AsyncFaceRecognizer:
    """Performs face recognition in a background thread using a persistent HTTP session."""

    def __init__(self, api_url: str, interval: float = 0.033, timeout: float = 3.0) -> None:
        self.api_url = api_url
        self.interval = interval
        self.timeout = timeout

        self._session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=1, pool_maxsize=2)
        self._session.mount("http://", adapter)

        self._lock = threading.Lock()
        self._latest_frame: Any | None = None
        self._faces: list[dict[str, Any]] = []
        self._error_msg: str | None = None
        self._api_latency: float = 0.0
        self._seq_id: int = 0
        self._ai_fps: float = 0.0
        self._frames_done: int = 0
        self._last_fps_time: float = time.monotonic()

        self._stop_event = threading.Event()
        self._worker_thread = threading.Thread(target=self._worker, daemon=True)

    def start(self) -> None:
        """Start the background recognition thread."""
        self._worker_thread.start()

    def stop(self) -> None:
        """Signal the background thread to stop and wait for it."""
        self._stop_event.set()
        if self._worker_thread.is_alive():
            self._worker_thread.join(timeout=1.0)
        self._session.close()

    def submit_frame(self, frame: Any) -> None:
        """Update the latest frame to be sent to the API."""
        with self._lock:
            self._latest_frame = frame

    def get_state(self) -> tuple[list[dict[str, Any]], str | None, float, int, float]:
        """Return the current recognized faces, last error, API latency, sequence id, and AI FPS."""
        with self._lock:
            return list(self._faces), self._error_msg, self._api_latency, self._seq_id, self._ai_fps

    def get_hw_info(self) -> str:
        return "HTTP Server"

    def _worker(self) -> None:
        """Background loop continuously sending the latest frame to the API."""
        while not self._stop_event.is_set():
            frame_to_process = None
            with self._lock:
                if self._latest_frame is not None:
                    frame_to_process = self._latest_frame
                    self._latest_frame = None

            if frame_to_process is not None:
                start_time = time.monotonic()
                try:
                    h, w = frame_to_process.shape[:2]
                    target_w = 320
                    if w > target_w:
                        scale = target_w / w
                        send_frame = cv2.resize(
                            frame_to_process,
                            (target_w, int(h * scale)),
                            interpolation=cv2.INTER_LINEAR,
                        )
                        inv_scale = w / target_w
                    else:
                        send_frame = frame_to_process
                        inv_scale = 1.0

                    ok, encoded = cv2.imencode(".jpg", send_frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                    if ok:
                        resp = self._session.post(
                            self.api_url,
                            files={"file": ("frame.jpg", encoded.tobytes(), "image/jpeg")},
                            timeout=self.timeout,
                        )
                        latency = time.monotonic() - start_time
                        if resp.status_code == 200:
                            payload = resp.json()
                            faces = payload.get("faces", [])
                            if inv_scale != 1.0:
                                for f in faces:
                                    bbox = f.get("bbox")
                                    if bbox and len(bbox) == 4:
                                        f["bbox"] = [c * inv_scale for c in bbox]
                            self._frames_done += 1
                            t_now = time.monotonic()
                            if t_now - self._last_fps_time >= 1.0:
                                self._ai_fps = self._frames_done / (t_now - self._last_fps_time)
                                self._frames_done = 0
                                self._last_fps_time = t_now
                            with self._lock:
                                self._faces = faces
                                self._error_msg = None
                                self._api_latency = latency
                                self._seq_id += 1
                        else:
                            with self._lock:
                                self._error_msg = f"HTTP {resp.status_code}"
                                self._api_latency = latency
                except Exception as exc:
                    latency = time.monotonic() - start_time
                    with self._lock:
                        self._error_msg = str(exc)
                        self._api_latency = latency

            if self.interval > 0:
                self._stop_event.wait(self.interval)


class InProcessFaceRecognizer:
    """Zero-Latency In-Memory Edge AI Recognizer.

    Khoảng cách kết nối bằng 0 (Zero-Latency In-Process):
    - Triệt tiêu nén JPEG (cv2.imencode) = 0ms
    - Triệt tiêu độ trễ mạng HTTP/TCP localhost = 0ms
    - Triệt tiêu giải mã JPEG (cv2.imdecode) = 0ms
    - Đưa trực tiếp mảng pixel NumPy raw trong bộ nhớ RAM vào InsightFace ONNX Engine.
    """

    def __init__(self, interval: float = 0.001, threshold: float | None = None) -> None:
        from app.config import SIMILARITY_THRESHOLD
        from app.core.face_engine import get_face_engine
        from app.core.tracker import FaceStreamTracker

        self.interval = interval
        self.threshold = threshold if threshold is not None else SIMILARITY_THRESHOLD
        self._engine = get_face_engine()
        self._tracker = FaceStreamTracker(re_id_interval=45)

        self._lock = threading.Lock()
        self._latest_frame: Any | None = None
        self._faces: list[dict[str, Any]] = []
        self._error_msg: str | None = None
        self._api_latency: float = 0.0
        self._seq_id: int = 0
        self._ai_fps: float = 0.0
        self._frames_done: int = 0
        self._last_fps_time: float = time.monotonic()

        self._stop_event = threading.Event()
        self._worker_thread = threading.Thread(target=self._worker, daemon=True)

    def start(self) -> None:
        """Start the background recognition thread."""
        self._worker_thread.start()

    def stop(self) -> None:
        """Signal the background thread to stop and wait for it."""
        self._stop_event.set()
        if self._worker_thread.is_alive():
            self._worker_thread.join(timeout=1.0)

    def submit_frame(self, frame: Any) -> None:
        """Submit the latest raw camera frame for recognition."""
        with self._lock:
            self._latest_frame = frame

    def get_state(self) -> tuple[list[dict[str, Any]], str | None, float, int, float]:
        """Return the current recognized faces, last error, inference latency, sequence id, and AI FPS."""
        with self._lock:
            return list(self._faces), self._error_msg, self._api_latency, self._seq_id, self._ai_fps

    def get_hw_info(self) -> str:
        dev = getattr(self._engine, "device_used", "CPU")
        precision = "INT8" if getattr(self._engine, "is_int8", False) else "FP32"
        hw = "Intel iGPU" if dev.upper() == "GPU" else f"Intel {dev}"
        return f"{hw} {precision}"

    def _worker(self) -> None:
        """Background inference loop executing directly in-memory."""
        while not self._stop_event.is_set():
            frame_to_process = None
            with self._lock:
                if self._latest_frame is not None:
                    frame_to_process = self._latest_frame
                    self._latest_frame = None

            if frame_to_process is not None:
                start_time = time.monotonic()
                try:
                    h, w = frame_to_process.shape[:2]
                    target_w = 320
                    if w > target_w:
                        scale = target_w / w
                        send_frame = cv2.resize(
                            frame_to_process,
                            (target_w, int(h * scale)),
                            interpolation=cv2.INTER_LINEAR,
                        )
                        inv_scale = w / target_w
                    else:
                        send_frame = frame_to_process
                        inv_scale = 1.0

                    # Ultra-Fast Hybrid Tracking-by-Detection (SCRFD on every frame, ArcFace only when needed)
                    raw_faces = self._engine.process_frame_hybrid(
                        send_frame, self._tracker, self.threshold
                    )

                    faces: list[dict[str, Any]] = []
                    for face in raw_faces:
                        scaled_bbox = [c * inv_scale for c in face["bbox"]]
                        faces.append({
                            "bbox": scaled_bbox,
                            "name": face["name"],
                            "id": face["id"],
                            "distance": face["distance"],
                            "similarity": face["similarity"],
                            "track_id": face.get("track_id"),
                            "state": face.get("state", "UNKNOWN"),
                        })

                    latency = time.monotonic() - start_time
                    self._frames_done += 1
                    t_now = time.monotonic()
                    if t_now - self._last_fps_time >= 1.0:
                        self._ai_fps = self._frames_done / (t_now - self._last_fps_time)
                        self._frames_done = 0
                        self._last_fps_time = t_now
                    with self._lock:
                        self._faces = faces
                        self._error_msg = None
                        self._api_latency = latency
                        self._seq_id += 1
                except Exception as exc:
                    latency = time.monotonic() - start_time
                    with self._lock:
                        self._error_msg = str(exc)
                        self._api_latency = latency

            if self.interval > 0:
                self._stop_event.wait(self.interval)


class SmoothBoxTracker:
    """Smooths bounding box transitions using Exponential Moving Average (EMA) and Spatial Tracking."""

    def __init__(self, alpha: float = 0.50) -> None:
        self.alpha = alpha
        self.tracks: list[dict[str, Any]] = []

    @staticmethod
    def _calc_iou(b1: list[float], b2: list[float]) -> float:
        ix1 = max(b1[0], b2[0])
        iy1 = max(b1[1], b2[1])
        ix2 = min(b1[2], b2[2])
        iy2 = min(b1[3], b2[3])
        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter = iw * ih
        a1 = max(0.0, b1[2] - b1[0]) * max(0.0, b1[3] - b1[1])
        a2 = max(0.0, b2[2] - b2[0]) * max(0.0, b2[3] - b2[1])
        union = a1 + a2 - inter
        return inter / union if union > 0 else 0.0

    def update(
        self,
        target_faces: list[dict[str, Any]],
        frame: Any | None = None,
        recognizer: Any | None = None,
        seq_id: int = 0,
    ) -> list[dict[str, Any]]:
        smoothed_faces = []
        if not target_faces:
            self.tracks.clear()
            return []

        matched_tracks: set[int] = set()
        new_tracks: list[dict[str, Any]] = []

        for fi, face in enumerate(target_faces):
            bbox = face.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            curr_b = [float(x) for x in bbox]
            f_tid = face.get("track_id")

            best_ti = -1
            # 1. Match by persistent track_id if available
            if f_tid is not None:
                for ti, trk in enumerate(self.tracks):
                    if ti not in matched_tracks and trk.get("track_id") == f_tid:
                        best_ti = ti
                        break

            # 2. Fallback to IoU matching
            if best_ti < 0:
                best_iou = 0.20
                for ti, trk in enumerate(self.tracks):
                    if ti in matched_tracks:
                        continue
                    score = self._calc_iou(trk["bbox"], curr_b)
                    if score > best_iou:
                        best_iou = score
                        best_ti = ti

            if best_ti >= 0:
                matched_tracks.add(best_ti)
                trk = self.tracks[best_ti]
                trk_bbox = [
                    trk["bbox"][j] + self.alpha * (curr_b[j] - trk["bbox"][j])
                    for j in range(4)
                ]
            else:
                trk_bbox = list(curr_b)

            new_tracks.append({"bbox": trk_bbox, "track_id": f_tid})
            smooth_face = dict(face)
            smooth_face["bbox"] = trk_bbox
            smoothed_faces.append(smooth_face)

        self.tracks = new_tracks
        return smoothed_faces


def draw_faces(frame: Any, faces: list[dict[str, Any]], threshold: float = 0.50) -> None:
    """Draw futuristic HUD corners and identity labels on the frame with 3 ratio tiers."""
    threshold_verified = float(threshold)
    threshold_holding = float(threshold) * (2.0 / 3.0)

    for face in faces:
        bbox = face.get("bbox", [])
        if len(bbox) != 4:
            continue
        x1, y1, x2, y2 = (int(value) for value in bbox)
        name = str(face.get("name") or "Unknown")
        distance = face.get("distance")
        similarity = face.get("similarity")

        state = face.get("state")
        sim_val = similarity if similarity is not None else (max(0.0, 1.0 - float(distance)) if distance is not None else 0.0)
        dist_val = f"{float(distance):.3f}" if distance is not None else "--"

        # Phân chia 3 khoảng mức tương đồng dựa trên 3 tỷ lệ của ngưỡng Threshold (T):
        # 1. Mức Chuẩn: VERIFIED (>= T)
        # 2. Mức Vừa: HOLDING ((2/3)*T <= sim < T)
        # 3. Mức Thấp: UNKNOWN (< (2/3)*T)
        has_id = bool(face.get("id") and face.get("name") != "Unknown")
        is_holding = (state in ("HOLDING", "COASTING") or face.get("matched") == 2 or (has_id and sim_val >= threshold_holding and sim_val < threshold_verified))
        is_verified = (state == "VERIFIED" or (has_id and sim_val >= threshold_verified and not is_holding))

        if is_verified:
            box_color = (0, 255, 157)  # BGR: Neon green (Mức Chuẩn)
            label = f"[XÁC THỰC CHUẨN] {name.upper()} (ID: {face.get('id', '?')})"
            conf_text = f"SIM: {sim_val * 100.0:.1f}% [CHUẨN >= {threshold_verified * 100:.0f}%]"
        elif is_holding:
            box_color = (0, 165, 255)  # BGR: Amber gold for HOLDING (Mức Vừa)
            cand_name = name.upper() if name and name != "Unknown" else "ĐANG TÁI XÁC THỰC"
            label = f"[HOLDING // TẠM GIỮ] {cand_name}"
            conf_text = f"SIM: {sim_val * 100.0:.1f}% [VỪA >= {threshold_holding * 100:.0f}%]"
        else:
            box_color = (85, 0, 255)   # BGR: Neon pink (Mức Thấp)
            label = "UNKNOWN SUBJECT"
            conf_text = f"SIM: {sim_val * 100.0:.1f}% [THẤP < {threshold_holding * 100:.0f}%]"

        # Corner lines
        arm = min(22, max(10, int((x2 - x1) * 0.2)))
        cv2.line(frame, (x1, y1), (x1 + arm, y1), box_color, 2)
        cv2.line(frame, (x1, y1), (x1, y1 + arm), box_color, 2)
        cv2.line(frame, (x2, y1), (x2 - arm, y1), box_color, 2)
        cv2.line(frame, (x2, y1), (x2, y1 + arm), box_color, 2)
        cv2.line(frame, (x1, y2), (x1 + arm, y2), box_color, 2)
        cv2.line(frame, (x1, y2), (x1, y2 - arm), box_color, 2)
        cv2.line(frame, (x2, y2), (x2 - arm, y2), box_color, 2)
        cv2.line(frame, (x2, y2), (x2, y2 - arm), box_color, 2)

        # Subtle bounding rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 1)

        # Label badge
        badge_w = max(200, int((x2 - x1) * 0.9))
        cv2.rectangle(frame, (x1, max(0, y1 - 28)), (x1 + badge_w, y1), (20, 20, 20), -1)
        cv2.rectangle(frame, (x1, max(0, y1 - 28)), (x1 + badge_w, y1), box_color, 1)
        cv2.putText(
            frame,
            f"{label} [{conf_text}]",
            (x1 + 6, max(15, y1 - 9)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            box_color,
            1,
            cv2.LINE_AA,
        )


def main() -> None:
    """Open the webcam and stream recognition at 30-60 FPS."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["in-process", "http"],
        default="in-process",
        help="Recognition mode: 'in-process' (Zero-Latency In-Memory Edge AI) or 'http' (FastAPI REST Client).",
    )
    parser.add_argument(
        "--api-url",
        default="http://127.0.0.1:8000/api/v1/recognize",
        help="Recognition endpoint URL (used when --mode http).",
    )
    parser.add_argument("--camera", type=int, default=0, help="Webcam index.")
    parser.add_argument(
        "--interval",
        type=float,
        default=0.000,
        help="Seconds between recognition frames (0.000 = Unconstrained max GPU throughput).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.50,
        help="Recognition minimum cosine similarity threshold (0.0 to 1.0, default: 0.50 = 50%% match).",
    )
    args = parser.parse_args()

    camera = ThreadedCamera(args.camera)
    if not camera.isOpened():
        raise RuntimeError("Cannot open webcam. Check camera permissions or index.")

    if args.mode == "in-process":
        print("=" * 65)
        print(" [ZERO-LATENCY MODE] In-Memory Edge AI Engine Active (0ms Network)")
        print(f" Running directly inside RAM with InsightFace + ChromaDB (Threshold: {args.threshold})")
        print("=" * 65)
        recognizer = InProcessFaceRecognizer(
            interval=max(0.0, args.interval),
            threshold=args.threshold,
        )
    else:
        api_url = args.api_url
        if "threshold=" not in api_url:
            sep = "&" if "?" in api_url else "?"
            api_url = f"{api_url}{sep}threshold={args.threshold}"
        print("=" * 65)
        print(f" [HTTP CLIENT MODE] Connecting to API: {api_url}")
        print("=" * 65)
        recognizer = AsyncFaceRecognizer(
            api_url=api_url,
            interval=max(0.0, args.interval),
            timeout=3.0,
        )

    recognizer.start()
    tracker = SmoothBoxTracker(alpha=0.50)

    print(" Press 'q' or ESC in the video window to quit.")
    print("=" * 65)

    fps = 0.0
    prev_time = time.monotonic()

    # Enable ultra-fine 1ms Windows timer resolution
    _has_winmm = False
    if sys.platform == "win32":
        try:
            ctypes.windll.winmm.timeBeginPeriod(1)
            _has_winmm = True
        except Exception:
            pass

    try:
        while True:
            ok, frame = camera.read()
            if not ok or frame is None:
                time.sleep(0.001)
                continue

            now = time.monotonic()
            dt = now - prev_time
            prev_time = now
            if dt > 0:
                current_fps = 1.0 / dt
                fps = fps * 0.9 + current_fps * 0.1

            # Submit frame to worker
            recognizer.submit_frame(frame)

            # Get latest faces & smooth them
            raw_faces, error_msg, latency, seq_id, ai_fps = recognizer.get_state()
            smooth_faces = tracker.update(raw_faces, frame=frame, recognizer=recognizer, seq_id=seq_id)

            # Draw
            draw_faces(frame, smooth_faces, threshold=args.threshold)

            # Telemetry overlay
            fps_color = (0, 255, 157) if fps >= 25 else (0, 165, 255)
            cv2.putText(
                frame,
                f"DISPLAY FPS: {fps:.0f} | AI INFERENCE FPS: {ai_fps:.0f} ({latency * 1000:.1f}ms)",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                fps_color,
                2,
                cv2.LINE_AA,
            )

            if latency > 0:
                hw_info = getattr(recognizer, "get_hw_info", lambda: "")()
                net_tag = f"0ms NET | {hw_info}" if args.mode == "in-process" else f"HTTP | {hw_info}"
                cv2.putText(
                    frame,
                    f"BACKEND: {hw_info} [{net_tag}]",
                    (15, 58),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    (0, 255, 157) if args.mode == "in-process" else (255, 240, 0),
                    1,
                    cv2.LINE_AA,
                )

            mode_text = "MODE: ZERO-LATENCY IN-MEMORY (RAM)" if args.mode == "in-process" else "MODE: HTTP REST CLIENT"
            cv2.putText(
                frame,
                mode_text,
                (15, 84),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 157) if args.mode == "in-process" else (0, 243, 255),
                1,
                cv2.LINE_AA,
            )

            if error_msg:
                cv2.putText(
                    frame,
                    f"STATUS: {error_msg[:35]}",
                    (15, 108),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 0, 255),
                    1,
                    cv2.LINE_AA,
                )

            cv2.imshow("NEURAL//VERIFY - Real-Time Stream", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break
    finally:
        recognizer.stop()
        camera.release()
        cv2.destroyAllWindows()
        if _has_winmm:
            try:
                ctypes.windll.winmm.timeEndPeriod(1)
            except Exception:
                pass


if __name__ == "__main__":
    main()

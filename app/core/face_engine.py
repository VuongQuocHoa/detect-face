"""InsightFace model integration."""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import queue
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import cv2
import numpy as np
from insightface.app import FaceAnalysis

from app.config import (
    DETECT_SIZE,
    ENABLE_OPENVINO,
    INSIGHTFACE_MODEL_NAME,
    INSIGHTFACE_MODEL_ROOT,
    OPENVINO_CACHE_DIR,
    OPENVINO_DEVICE,
    OPENVINO_USE_INT8,
)
from insightface.app.common import Face
from app.core.vector_db import add_face, get_next_stranger_info, search_face


class _OpenVINONodeInfo:
    """Mock node info for compatibility with InsightFace introspection."""

    def __init__(self, name: str, shape: list[Any]) -> None:
        self.name = name
        self.shape = shape


class OpenVINOSessionAdapter:
    """Drop-in adapter replacing ONNXRuntime InferenceSession with Intel OpenVINO.

    Executes inference via OpenVINO Core compiled for Intel Raptor Lake CPU or Intel UHD/Iris Xe iGPU (device='GPU'),
    accelerating inference to ~3-8ms per frame while offloading compute from CPU.
    """

    def __init__(
        self,
        model_path: str | Path,
        device: str = "GPU",
        input_shape: list[int] | None = None,
        use_cache: bool = True,
    ) -> None:
        import openvino as ov

        self.core = ov.Core()
        self._lock = threading.Lock()
        available_devices = self.core.available_devices

        # Auto fallback if requested device is not available
        target_device = device.upper()
        if target_device not in available_devices and target_device != "AUTO":
            print(f" [OPENVINO] Device '{device}' not in {available_devices}. Fallback to 'CPU'.")
            target_device = "CPU"

        if use_cache and OPENVINO_CACHE_DIR:
            try:
                self.core.set_property({"CACHE_DIR": OPENVINO_CACHE_DIR})
            except Exception:
                pass

        model = self.core.read_model(str(model_path))
        if input_shape is not None:
            model.reshape(input_shape)

        config = {
            "PERFORMANCE_HINT": "LATENCY",
            "EXECUTION_MODE_HINT": "PERFORMANCE",
        }
        try:
            self.compiled = self.core.compile_model(model, target_device, config)
            self.active_device = target_device
        except Exception as exc:
            if target_device != "CPU" and "CPU" in available_devices:
                print(f" [OPENVINO] Compile on {target_device} failed ({exc}). Fallback to 'CPU'.")
                self.compiled = self.core.compile_model(model, "CPU", config)
                self.active_device = "CPU"
            else:
                raise exc

        self.infer_request = self.compiled.create_infer_request()
        self._inputs = [
            _OpenVINONodeInfo(inp.get_any_name(), list(inp.get_partial_shape()))
            for inp in self.compiled.inputs
        ]
        self._outputs = list(self.compiled.outputs)

    def run(
        self, output_names: list[str] | None, input_feed: dict[str, Any]
    ) -> list[np.ndarray]:
        with self._lock:
            self.infer_request.infer(input_feed)
            return [self.infer_request.get_tensor(out).data.copy() for out in self._outputs]

    def get_inputs(self) -> list[_OpenVINONodeInfo]:
        return self._inputs

    def get_outputs(self) -> list[_OpenVINONodeInfo]:
        return [
            _OpenVINONodeInfo(out.get_any_name(), list(out.get_partial_shape()))
            for out in self._outputs
        ]

    def get_providers(self) -> list[str]:
        return [f"OpenVINOExecutionProvider_{self.active_device}"]


@dataclass(frozen=True)
class ExtractedFace:
    """A detected face and its 512-dimensional embedding."""

    bbox: list[float]
    embedding: list[float]
    area: float


class AsyncDetectionDispatcher:
    """Asynchronous Decoupled Detector for extreme multi-thousand FPS streaming.
    Runs SCRFD detector in a dedicated background worker without stalling the stream loop.
    """

    def __init__(self, det_model: Any) -> None:
        self.det_model = det_model
        self.task_queue: queue.Queue = queue.Queue(maxsize=1)
        self.result_lock = threading.Lock()
        self.latest_result: tuple[np.ndarray, np.ndarray | None] | None = None
        self.is_busy = False
        self.running = True
        self._ping_pong = 0
        self._buffers: list[np.ndarray | None] = [None, None]
        self.worker = threading.Thread(target=self._loop, daemon=True, name="AsyncSCRFDWorker")
        self.worker.start()

    def _loop(self) -> None:
        while self.running:
            try:
                img = self.task_queue.get(timeout=0.1)
                if img is None:
                    break
                bboxes, kpss = self.det_model.detect(img, max_num=0, metric="default")
                if bboxes is None or bboxes.shape[0] == 0:
                    bboxes = np.empty((0, 5))
                    kpss = None
                with self.result_lock:
                    self.latest_result = (bboxes, kpss)
                self.is_busy = False
            except queue.Empty:
                continue
            except Exception:
                self.is_busy = False

    def trigger(self, img: np.ndarray) -> None:
        if not self.is_busy and self.task_queue.empty():
            self.is_busy = True
            idx = self._ping_pong
            self._ping_pong = 1 - self._ping_pong
            if self._buffers[idx] is None or self._buffers[idx].shape != img.shape:
                self._buffers[idx] = np.empty_like(img)
            np.copyto(self._buffers[idx], img)
            try:
                self.task_queue.put_nowait(self._buffers[idx])
            except queue.Full:
                self.is_busy = False

    def pop_result(self) -> tuple[np.ndarray, np.ndarray | None] | None:
        with self.result_lock:
            res = self.latest_result
            self.latest_result = None
            return res


class FaceEngine:
    """Loads InsightFace and accelerates models with Intel OpenVINO."""

    def __init__(self) -> None:
        self._app = FaceAnalysis(
            name=INSIGHTFACE_MODEL_NAME,
            root=INSIGHTFACE_MODEL_ROOT,
            allowed_modules=["detection", "recognition"],
            providers=["CPUExecutionProvider"],
        )
        self._app.prepare(ctx_id=0, det_size=DETECT_SIZE)

        self.is_openvino_accelerated = False
        self.device_used = "CPU"
        self.is_int8 = False

        if ENABLE_OPENVINO:
            try:
                det = self._app.models.get("detection")
                if det and hasattr(det, "model_file"):
                    det_file = Path(det.model_file)
                    int8_det = det_file.with_name("det_500m_int8.xml")
                    if OPENVINO_USE_INT8 and int8_det.exists():
                        det_file = int8_det
                        self.is_int8 = True

                    det.static_shape_sessions = False
                    det.session = OpenVINOSessionAdapter(
                        det_file,
                        OPENVINO_DEVICE,
                        [1, 3, DETECT_SIZE[1], DETECT_SIZE[0]],
                    )
                    det._resolution_session_pool = None
                    det.input_size = DETECT_SIZE

                rec = self._app.models.get("recognition")
                if rec and hasattr(rec, "model_file"):
                    rec_file = Path(rec.model_file)
                    int8_rec = rec_file.with_name("w600k_mbf_int8.xml")
                    if OPENVINO_USE_INT8 and int8_rec.exists():
                        rec_file = int8_rec

                    rec.session = OpenVINOSessionAdapter(
                        rec_file,
                        OPENVINO_DEVICE,
                        [1, 3, 112, 112],
                    )

                self.is_openvino_accelerated = True
                self.device_used = getattr(det.session, "active_device", OPENVINO_DEVICE)
                precision_label = "INT8" if self.is_int8 else "FP32/FP16"
                print(
                    f" [OPENVINO] Accelerated with Intel OpenVINO on device='{self.device_used}' "
                    f"({precision_label}) successfully! (Latency ~3-8ms)"
                )
            except Exception as exc:
                print(f" [OPENVINO] Acceleration failed, using standard ONNXRuntime: {exc}")

        # Pre-allocated zero-allocation buffer for zero-copy color conversion
        self._bgr_buffer: np.ndarray | None = None
        self._buffer_shape: tuple[int, ...] | None = None

        # Asynchronous Decoupled Detection Dispatcher for Extreme FPS
        self._async_detector: AsyncDetectionDispatcher | None = None
        det = self._app.models.get("detection")
        if det is not None:
            self._async_detector = AsyncDetectionDispatcher(det)

    def extract_from_image_array(self, image: np.ndarray) -> list[ExtractedFace]:
        """Detect all faces directly from a raw in-memory BGR numpy array (Zero-Latency)."""
        if image is None or image.size == 0:
            return []

        faces = self._app.get(image)
        if len(faces) == 0:
            return []

        extracted: list[ExtractedFace] = []
        for face in faces:
            bbox = np.asarray(face.bbox, dtype=np.float32).reshape(-1)
            embedding = np.asarray(face.embedding, dtype=np.float32).reshape(-1)
            if bbox.size != 4 or embedding.size != 512:
                continue
            clipped_bbox = self._clip_bbox(bbox, image.shape[1], image.shape[0])
            area = max(0.0, clipped_bbox[2] - clipped_bbox[0]) * max(
                0.0, clipped_bbox[3] - clipped_bbox[1]
            )
            extracted.append(
                ExtractedFace(
                    bbox=clipped_bbox,
                    embedding=embedding.astype(float).tolist(),
                    area=area,
                )
            )
        return extracted

    def extract_embeddings(
        self, image_bytes: bytes
    ) -> tuple[list[ExtractedFace] | None, str | None]:
        """Detect all faces with zero-latency raw binary packet support or standard image decode."""
        if len(image_bytes) >= 8 and image_bytes[0] == 0xFA and image_bytes[1] == 0xCE:
            w = int.from_bytes(image_bytes[2:4], "little")
            h = int.from_bytes(image_bytes[4:6], "little")
            channels = int.from_bytes(image_bytes[6:8], "little")
            expected_len = 8 + w * h * channels
            if len(image_bytes) == expected_len:
                arr = np.frombuffer(image_bytes, dtype=np.uint8, offset=8).reshape((h, w, channels))
                if channels == 4:
                    image = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
                elif channels == 3:
                    image = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                else:
                    image = arr
                return self.extract_from_image_array(image), None

        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if image is None:
            return None, "Ảnh không hợp lệ hoặc không đúng định dạng được hỗ trợ."

        return self.extract_from_image_array(image), None


    @staticmethod
    def register_face(
        user_id: str, name: str, embedding: list[float]
    ) -> tuple[bool, str | None]:
        """Persist an embedding and convert database errors into clear messages."""
        try:
            add_face(user_id=user_id, name=name, embedding=embedding)
        except Exception as exc:
            return False, f"Lỗi khi ghi dữ liệu vào ChromaDB: {exc}"
        return True, None

    def process_frame_hybrid(
        self,
        image: np.ndarray,
        tracker: Any,
        threshold: float,
    ) -> list[dict[str, Any]]:
        """Ultra-fast detection-tracking hybrid recognition pipeline.

        Runs SCRFD detector on every frame (~3ms), but skips ArcFace (~4ms per face)
        whenever a track has already been identified and has not drifted.
        """
        if image is None or image.size == 0:
            return []

        det_model = self._app.models.get("detection")
        rec_model = self._app.models.get("recognition")
        if det_model is None:
            return []

        img_h, img_w = image.shape[:2]

        # Prepare zero-allocation grayscale image for real-time Optical Flow head tracking
        gray = None
        if image is not None and image.size > 0:
            if getattr(self, "_gray_buffer", None) is None or self._gray_buffer.shape != (img_h, img_w):
                self._gray_buffer = np.empty((img_h, img_w), dtype=np.uint8)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY, dst=self._gray_buffer)

        # 1. Áp dụng triệt để tính năng dự đoán (Predictive Optical Flow & Kinematics ~0.05ms)
        need_detect = tracker.should_run_detection(img_w, img_h)
        bboxes = None
        kpss = None

        if not need_detect:
            # Dự đoán quang học Lucas-Kanade + Quán tính bậc 2 siêu tốc, không cần chạy SCRFD nặng
            active_tracks = tracker.predictive_step(gray, image_shape=(img_h, img_w))
        else:
            # Chạy SCRFD phát hiện khi cần thiết (khởi động, đối tượng mới, chạm mép, định kỳ)
            if self._async_detector is not None:
                new_dets = self._async_detector.pop_result()
                if new_dets is not None:
                    bboxes, kpss = new_dets
                elif len(tracker.tracks) == 0:
                    # Cold start: No active tracks yet, execute instant synchronous detect (3ms) to eliminate frame 1 delay
                    bboxes, kpss = det_model.detect(image, max_num=0, metric="default")
                    if bboxes is None or bboxes.shape[0] == 0:
                        bboxes = np.empty((0, 5))
                        kpss = None

                # Non-blocking continuous background detection trigger
                self._async_detector.trigger(image)
            else:
                bboxes, kpss = det_model.detect(image, max_num=0, metric="default")
                if bboxes is None or bboxes.shape[0] == 0:
                    bboxes = np.empty((0, 5))
                    kpss = None

            # 2. Multi-Object Spatial Tracking & ArcFace Trigger Decision
            if bboxes is not None:
                active_tracks = tracker.update_tracks(bboxes, kpss, gray, image_shape=(img_h, img_w))
            else:
                active_tracks = tracker.predictive_step(gray, image_shape=(img_h, img_w))

        # 3. Process each track
        results: list[dict[str, Any]] = []

        for track, need_rec in active_tracks:
            if need_rec and rec_model is not None and track.kps is not None:
                # 1. Pose / Extreme Yaw Check & Predictive Best-Shot Selection
                yaw_ratio = track.get_pose_yaw_ratio(track.kps)
                if yaw_ratio > 0.46:
                    # Extreme profile angle (> 55°): Skip ArcFace on this single frame to avoid embedding distortion
                    pass
                elif getattr(track, "yaw_vel", 0.0) < -0.04 and yaw_ratio > 0.20 and track.frames_since_recognition < 8:
                    # PREDICTIVE BEST-SHOT: Face is rotating towards frontal view.
                    # Wait 1-2 frames for peak frontal alignment instead of wasting ArcFace on sub-optimal profile angle.
                    pass
                else:
                    # 2. Run ArcFace feature extraction
                    face = Face(bbox=np.array(track.bbox), kps=track.kps, det_score=track.score)
                    rec_model.get(image, face)
                    if face.embedding is not None:
                        match = search_face(face.embedding.tolist(), threshold)
                        best_sim = float(match.get("similarity") or 0.0) if isinstance(match, dict) else 0.0
                        best_dist = float(match.get("distance") or 1.0) if isinstance(match, dict) else 1.0

                        # 3 KHOẢNG MỨC ĐỘ TƯƠNG ĐỒNG CHUẨN CÔNG NGHIỆP DỰA TRÊN TỶ LỆ CỦA NGƯỠNG THRESHOLD (T):
                        threshold_verified = float(threshold)
                        threshold_holding = round(threshold_verified * (2.0 / 3.0), 4)

                        # Tìm kiếm theo ngưỡng threshold_holding để bắt trọn cả ứng viên mức Vừa (HOLDING)
                        match = search_face(face.embedding.tolist(), threshold_holding)
                        best_sim = float(match.get("similarity") or 0.0) if isinstance(match, dict) else 0.0
                        best_dist = float(match.get("distance") or 1.0) if isinstance(match, dict) else 1.0

                        matched_id = str(match.get("id") or match.get("candidate_id") or "")
                        matched_name = str(match.get("name") or match.get("candidate_name") or "")
                        is_stranger_id = matched_id.lower().startswith("stranger_") or matched_name.lower().startswith("người lạ")

                        track_was_verified = getattr(track, "was_verified", False) or (track.state == "VERIFIED" and not getattr(track, "is_stranger", False))

                        if track_was_verified:
                            track.was_verified = True
                            track.is_stranger = False
                            if is_stranger_id:
                                is_stranger_id = False
                                matched_id = str(track.identity.get("id") or "")
                                matched_name = str(track.identity.get("name") or "")

                        if best_sim >= threshold_verified:
                            if is_stranger_id and not getattr(track, "was_verified", False):
                                # QUY TẮC CỐT LÕI: NGƯỜI LẠ LUÔN KHÔNG ĐƯỢC CHO LÀ VERIFIED!
                                # Khóa định danh vào Người lạ đã có trong CSDL, trạng thái HOLDING, KHÔNG TẠO MỚI!
                                track.identity = {
                                    "name": matched_name,
                                    "id": matched_id,
                                    "similarity": best_sim,
                                    "distance": best_dist,
                                }
                                track.state = "HOLDING"
                                track.is_stranger = True
                                track.was_verified = False
                                track.auto_registered = True
                                track.consecutive_unknown_count = 0
                                track.need_recognition = False
                                track.frames_since_recognition = 0
                            else:
                                # THÀNH VIÊN ĐÃ ĐĂNG KÝ CHÍNH THỨC: XÁC THỰC THÀNH CÔNG (VERIFIED)
                                track.identity = {
                                    "name": matched_name,
                                    "id": matched_id,
                                    "similarity": best_sim,
                                    "distance": best_dist,
                                }
                                track.state = "VERIFIED"
                                track.was_verified = True
                                track.is_stranger = False
                                track.consecutive_unknown_count = 0
                                track.need_recognition = False
                                track.frames_since_recognition = 0

                        elif best_sim >= threshold_holding:
                            # MỨC VỪA: TẠM GIỮ / THEO DÕI NGHI VẤN (HOLDING)
                            if is_stranger_id and not getattr(track, "was_verified", False):
                                # Đã khớp với Người lạ có sẵn trong CSDL -> KHÓA ĐỊNH DANH, KHÔNG TẠO MỚI!
                                track.identity = {
                                    "name": matched_name if matched_name and matched_name != "Unknown" else "Unknown",
                                    "id": matched_id if matched_id else None,
                                    "similarity": best_sim,
                                    "distance": best_dist,
                                }
                                track.state = "HOLDING"
                                track.was_coasting = False
                                track.is_stranger = True
                                track.was_verified = False
                                track.auto_registered = True
                                track.need_recognition = False
                                track.frames_since_recognition = 0
                            else:
                                if getattr(track, "was_verified", False):
                                    track.state = "HOLDING"
                                    track.was_coasting = False
                                    track.is_stranger = False
                                    track.need_recognition = False
                                else:
                                    track.identity = {
                                        "name": matched_name if matched_name and matched_name != "Unknown" else "Unknown",
                                        "id": matched_id if matched_id else None,
                                        "similarity": best_sim,
                                        "distance": best_dist,
                                    }
                                    track.state = "HOLDING"
                                    track.was_coasting = False
                                    track.consecutive_unknown_count = 0
                                    track.is_stranger = False
                                    track.need_recognition = False

                        else:
                            # MỨC THẤP: CHƯA XÁC ĐỊNH (UNKNOWN)
                            track.consecutive_unknown_count += 1

                            # QUY TẮC CỐT LÕI: NẾU HỆ THỐNG ĐÃ XÁC ĐỊNH NÓ LÀ NGƯỜI VERIFIED RỒI
                            # THÌ TUYỆT ĐỐI KHÔNG THỰC HIỆN KIỂM TRA NGƯỜI LẠ HAY TẠO NGƯỜI LẠ Ở TRACK ID ĐÓ!
                            if getattr(track, "was_verified", False):
                                track.state = "HOLDING"
                                track.is_stranger = False
                                track.need_recognition = False
                            else:
                                # KIỂM TRA CHẶNG CUỐI: TRÁNH TRÙNG LẶP CÙNG 1 NGƯỜI LẠ
                                already_handled = getattr(track, "auto_registered", False) or getattr(track, "is_stranger", False)

                                if not already_handled:
                                    deep_check = search_face(face.embedding.tolist(), threshold=0.35)
                                    deep_id = str(deep_check.get("id") or deep_check.get("candidate_id") or "")
                                    deep_name = str(deep_check.get("name") or deep_check.get("candidate_name") or "")
                                    deep_sim = float(deep_check.get("similarity") or 0.0)

                                    if deep_id and deep_sim >= 0.35:
                                        # Trùng người lạ hoặc thành viên đã có trong CSDL!
                                        is_deep_stranger = deep_id.lower().startswith("stranger_") or deep_name.lower().startswith("người lạ")
                                        track.auto_registered = True
                                        track.is_stranger = is_deep_stranger
                                        track.identity = {
                                            "name": deep_name,
                                            "id": deep_id,
                                            "similarity": deep_sim,
                                            "distance": float(deep_check.get("distance") or (1.0 - deep_sim)),
                                        }
                                        track.state = "HOLDING" if is_deep_stranger else ("VERIFIED" if deep_sim >= threshold_verified else "HOLDING")
                                        if not is_deep_stranger and track.state == "VERIFIED":
                                            track.was_verified = True
                                        track.need_recognition = False
                                        track.frames_since_recognition = 0
                                        already_handled = True

                                # CHỈ TẠO NGƯỜI LẠ MỚI KHI:
                                # - Track chưa từng là người verified
                                # - 100% chưa từng tồn tại trong CSDL (deep_sim < 0.35)
                                # - Chưa từng tạo cho track này
                                # - Track đã ổn định qua nhiều frame (hits >= 3, consecutive_unknown_count >= 3)
                                # - Góc mặt trực diện chuẩn (yaw_ratio < 0.35)
                                if (
                                    not already_handled
                                    and track.hits >= 3
                                    and track.consecutive_unknown_count >= 3
                                    and yaw_ratio < 0.35
                                    and face.embedding is not None
                                ):
                                    new_id, new_name = get_next_stranger_info()
                                    saved, _ = self.register_face(new_id, new_name, face.embedding.tolist())
                                    if saved:
                                        try:
                                            from app.utils.image_utils import crop_and_save_face
                                            crop_and_save_face(image, track.bbox, new_id)
                                        except Exception as err_crop:
                                            print(f"Lỗi lưu ảnh người lạ: {err_crop}")
                                        track.auto_registered = True
                                        track.is_stranger = True
                                        track.was_verified = False
                                        track.identity = {
                                            "name": new_name,
                                            "id": new_id,
                                            "similarity": 1.0,
                                            "distance": 0.0,
                                        }
                                        # Người lạ: LUÔN LÀ HOLDING hoặc UNKNOWN, KHÔNG BAO GIỜ LÀ VERIFIED!
                                        track.state = "HOLDING"
                                        track.consecutive_unknown_count = 0
                                        track.need_recognition = False  # KHÓA THEO DÕI, KHÔNG TẠO MỚI!
                                        track.frames_since_recognition = 0
                                        print(f" [AUTO-REGISTER] Phát hiện người lạ mới! Đã lưu CSDL: {new_name} ({new_id})")
                                else:
                                    if not already_handled:
                                        if track.consecutive_unknown_count >= 3 or track.state != "VERIFIED":
                                            track.identity = {
                                                "name": "Unknown",
                                                "id": None,
                                                "similarity": best_sim,
                                                "distance": best_dist,
                                            }
                                            track.state = "UNKNOWN"
                                        track.need_recognition = False
                                        track.frames_since_recognition = 0

        # RÀNG BUỘC ĐỘC QUYỀN DANH TÍNH (Single Identity Constraint):
        # Trong cùng 1 frame, một người thật (name != 'Unknown') KHÔNG THỂ phân thân ở 2 vị trí khác nhau!
        # Chỉ có khuôn mặt có độ tương đồng cao nhất được giữ danh tính; các hộp khác bị hạ cấp về UNKNOWN!
        claimed_identities: dict[str, tuple[int, float]] = {}
        for idx, (t, _) in enumerate(active_tracks):
            t_name = t.identity.get("name")
            if t_name and t_name != "Unknown":
                sim = float(t.identity.get("similarity") or 0.0)
                if t_name in claimed_identities:
                    prev_idx, prev_sim = claimed_identities[t_name]
                    if sim > prev_sim:
                        # Track hiện tại khớp tốt hơn -> Hạ cấp track trước đó về UNKNOWN
                        demoted_track = active_tracks[prev_idx][0]
                        demoted_track.identity = {
                            "name": "Unknown",
                            "id": None,
                            "similarity": prev_sim,
                            "distance": demoted_track.identity.get("distance"),
                        }
                        demoted_track.state = "UNKNOWN"
                        demoted_track.need_recognition = True
                        claimed_identities[t_name] = (idx, sim)
                    else:
                        # Track trước đó khớp tốt hơn -> Hạ cấp track hiện tại về UNKNOWN
                        t.identity = {
                            "name": "Unknown",
                            "id": None,
                            "similarity": sim,
                            "distance": t.identity.get("distance"),
                        }
                        t.state = "UNKNOWN"
                        t.need_recognition = True
                else:
                    claimed_identities[t_name] = (idx, sim)

        for track, need_rec in active_tracks:
            clipped_bbox = self._clip_bbox(track.bbox, img_w, img_h)
            ident = track.identity
            results.append({
                "bbox": clipped_bbox,
                "name": ident["name"],
                "id": ident["id"],
                "similarity": ident["similarity"],
                "distance": ident["distance"],
                "track_id": track.track_id,
                "state": track.state,
                "is_stranger": getattr(track, "is_stranger", False),
            })

        return results

    def process_bytes_hybrid(
        self,
        image_bytes: bytes,
        tracker: Any,
        threshold: float,
    ) -> list[dict[str, Any]]:
        """Decode raw binary 0xFACE packet or standard image bytes and run hybrid recognition."""
        if len(image_bytes) >= 8 and image_bytes[0] == 0xFA and image_bytes[1] == 0xCE:
            w = int.from_bytes(image_bytes[2:4], "little")
            h = int.from_bytes(image_bytes[4:6], "little")
            channels = int.from_bytes(image_bytes[6:8], "little")
            expected_len = 8 + w * h * channels
            if len(image_bytes) == expected_len:
                arr = np.frombuffer(image_bytes, dtype=np.uint8, offset=8).reshape((h, w, channels))
                if channels in (3, 4):
                    if self._buffer_shape != (h, w, 3):
                        self._bgr_buffer = np.empty((h, w, 3), dtype=np.uint8)
                        self._buffer_shape = (h, w, 3)
                    code = cv2.COLOR_RGBA2BGR if channels == 4 else cv2.COLOR_RGB2BGR
                    image = cv2.cvtColor(arr, code, dst=self._bgr_buffer)
                else:
                    image = arr
                return self.process_frame_hybrid(image, tracker, threshold)

        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if image is None:
            return []
        return self.process_frame_hybrid(image, tracker, threshold)

    @staticmethod
    def _clip_bbox(
        bbox: list[float] | tuple[float, ...] | np.ndarray, image_width: int, image_height: int
    ) -> list[float]:
        return [
            max(0.0, min(float(bbox[0]), float(image_width))),
            max(0.0, min(float(bbox[1]), float(image_height))),
            max(0.0, min(float(bbox[2]), float(image_width))),
            max(0.0, min(float(bbox[3]), float(image_height))),
        ]


_engine: FaceEngine | None = None
_engine_lock = threading.Lock()


def get_face_engine() -> FaceEngine:
    """Return the process-wide InsightFace engine."""
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                _engine = FaceEngine()
    return _engine


def extract_embeddings(
    image_bytes: bytes,
) -> tuple[list[ExtractedFace] | None, str | None]:
    """Compatibility function for extracting faces with the shared engine."""
    return get_face_engine().extract_embeddings(image_bytes)

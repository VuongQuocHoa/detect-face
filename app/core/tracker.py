"""Industrial-grade Spatial Face Tracker with Selective ArcFace Cache.

Solves the 4 classic edge-case pitfalls of Tracking-by-Detection:
1. Identity Swapping: Prevented via trajectory velocity prediction + proximity ambiguity trigger.
2. Occlusion & Pose Changes: Handled via track coasting, age tracking, and re-acquisition.
3. Multi-Object Tracking: Independent track IDs and greedy IoU matching.
4. False Negative / New Person: Any unmatched detection immediately triggers ArcFace on frame 1.
"""

import cv2
from typing import Any
import numpy as np


class FaceTrack:
    """Represents a continuous facial trajectory with Lucas-Kanade Optical Flow head tracking."""

    def __init__(
        self,
        track_id: int,
        bbox: list[float],
        kps: np.ndarray | None = None,
        score: float = 1.0,
    ) -> None:
        self.track_id = track_id
        self.bbox = [float(x) for x in bbox]  # [x1, y1, x2, y2]
        self.kps = kps
        self.score = score

        # Invariant reference head dimensions (locked to maximum verified head size)
        self.w = max(24.0, float(bbox[2] - bbox[0]))
        self.h = max(24.0, float(bbox[3] - bbox[1]))
        self.base_w = self.w
        self.base_h = self.h
        self.cx = float(bbox[0] + bbox[2]) * 0.5
        self.cy = float(bbox[1] + bbox[3]) * 0.5
        self.center_vel = [0.0, 0.0]  # [vx, vy]
        # Second-order kinematics: acceleration & scale velocity
        self.accel = [0.0, 0.0]       # [ax, ay]
        self.scale_vel = [0.0, 0.0]   # [vw, vh]
        self.last_yaw: float = 0.5
        self.yaw_vel: float = 0.0

        # Optical Flow Feature Anchors on head/hair
        self.features: np.ndarray | None = None
        self.initial_feature_count: int = 0

        self.age = 1
        self.hits = 1
        self.time_since_update = 0

        # Identity Cache
        self.identity: dict[str, Any] = {
            "name": "Unknown",
            "id": None,
            "similarity": None,
            "distance": None,
        }
        self.state: str = "UNKNOWN"  # "VERIFIED", "HOLDING", "UNKNOWN"
        self.is_stranger: bool = False
        self.was_verified: bool = False
        self.auto_registered: bool = False
        self.consecutive_unknown_count: int = 0
        self.was_coasting: bool = False
        self.need_recognition = True
        self.frames_since_recognition = 0
        self.still_frames = 0

    @staticmethod
    def get_pose_yaw_ratio(kps: np.ndarray | None) -> float:
        """Estimates facial yaw (profile rotation) from 5 landmarks (0.0=frontal, >0.45=extreme profile)."""
        if kps is None or len(kps) < 5:
            return 0.0
        left_eye, right_eye, nose = kps[0], kps[1], kps[2]
        eye_center_x = float(left_eye[0] + right_eye[0]) * 0.5
        eye_span = abs(float(right_eye[0] - left_eye[0]))
        if eye_span < 4.0:
            return 0.99  # Profile/extreme angle
        nose_offset = abs(float(nose[0]) - eye_center_x)
        return float(nose_offset / eye_span)

    def check_facial_completeness(self, db_bbox: list[float], kps: np.ndarray | None) -> bool:
        """Determines if a detected face is complete and unoccluded, or if hand/turn is occluding it."""
        curr_w = db_bbox[2] - db_bbox[0]
        curr_h = db_bbox[3] - db_bbox[1]

        # 1. Dimension test: If detected box shrunk > 18% compared to established base head dimensions,
        # it indicates a hand is covering the chin, mouth, or sides of face.
        ref_w = max(self.w, getattr(self, "base_w", self.w))
        ref_h = max(self.h, getattr(self, "base_h", self.h))
        if curr_w < ref_w * 0.82 or curr_h < ref_h * 0.82:
            return False

        # 2. Keypoints completeness test (5 landmarks: 2 eyes, 1 nose, 2 mouth corners)
        if kps is None or len(kps) < 5:
            return False

        # Check landmarks geometry: distance between eyes and mouth must be proportional to face height
        left_eye, right_eye, nose, left_mouth, right_mouth = kps[:5]
        eye_y = float(left_eye[1] + right_eye[1]) * 0.5
        mouth_y = float(left_mouth[1] + right_mouth[1]) * 0.5
        vertical_span = mouth_y - eye_y

        # Normal adult face: eye-to-mouth distance is ~30% - 65% of bounding box height.
        # If hand covers mouth/chin, mouth landmarks collapse upward or become unreliably close.
        if vertical_span < (curr_h * 0.28):
            return False

        return True

    def is_out_of_bounds(self, img_w: int, img_h: int) -> bool:
        """Checks if the tracked head has physically exited the camera viewport."""
        w_f = float(img_w)
        h_f = float(img_h)
        # 1. Center of head has left viewport
        if self.cx < 0.0 or self.cx > w_f or self.cy < 0.0 or self.cy > h_f:
            return True
        # 2. More than 75% of head is out of frame
        if (self.cx < self.w * 0.25) or (self.cx > w_f - self.w * 0.25):
            return True
        if (self.cy < self.h * 0.25) or (self.cy > h_f - self.h * 0.25):
            return True
        # 3. Box boundaries completely off-screen
        margin = 8.0
        if self.bbox[2] < margin or self.bbox[0] > (w_f - margin):
            return True
        if self.bbox[3] < margin or self.bbox[1] > (h_f - margin):
            return True
        return False

    def is_touching_edge(self, img_w: int, img_h: int, threshold: float = 6.0) -> bool:
        """Checks if the tracked bounding box touches or crosses any camera edge."""
        w_f = float(img_w)
        h_f = float(img_h)
        return (
            self.bbox[0] <= threshold or
            self.bbox[2] >= (w_f - threshold) or
            self.bbox[1] <= threshold or
            self.bbox[3] >= (h_f - threshold)
        )

    def predict(self) -> list[float]:
        """Second-order kinematic prediction with acceleration and scale expansion."""
        # Bậc 2: vị trí tương lai = x + v + 0.5 * a
        pred_cx = self.cx + self.center_vel[0] + 0.5 * self.accel[0]
        pred_cy = self.cy + self.center_vel[1] + 0.5 * self.accel[1]

        # Scale prediction: mở rộng/thu nhỏ mượt mà khi người tiến sát hoặc lùi xa camera
        pred_w = max(24.0, self.w + self.scale_vel[0])
        pred_h = max(24.0, self.h + self.scale_vel[1])
        hw = pred_w * 0.5
        hh = pred_h * 0.5
        return [pred_cx - hw, pred_cy - hh, pred_cx + hw, pred_cy + hh]

    def refresh_features(self, gray_image: np.ndarray | None) -> None:
        """Extract optical flow anchor points on perimeter of head (hair, forehead, temples, ears, outer jaw).
        
        Excludes the central face core (eyes, nose, mouth) so that when hands, phones, or masks
        cover the face, the perimeter anchor points remain undisturbed.
        """
        if gray_image is None or gray_image.size == 0:
            return
        img_h, img_w = gray_image.shape[:2]
        ref_w = max(self.w, getattr(self, "base_w", self.w))
        ref_h = max(self.h, getattr(self, "base_h", self.h))

        # Outer perimeter bounding box covering entire head, hairline, ears, and temples
        x1 = max(0, int(self.cx - ref_w * 0.70))
        y1 = max(0, int(self.cy - ref_h * 0.85))
        x2 = min(img_w, int(self.cx + ref_w * 0.70))
        y2 = min(img_h, int(self.cy + ref_h * 0.55))

        if x2 - x1 < 16 or y2 - y1 < 16:
            return

        patch_w = x2 - x1
        patch_h = y2 - y1
        mask = np.ones((patch_h, patch_w), dtype=np.uint8) * 255

        # Mask out center facial core (eyes, nose, mouth, chin center) to preserve perimeter corners
        center_x_in_patch = int(self.cx - x1)
        center_y_in_patch = int(self.cy - y1)
        ex_hw = int(ref_w * 0.32)
        ex_top = int(ref_h * 0.12)
        ex_bot = int(ref_h * 0.48)

        cx1 = max(0, center_x_in_patch - ex_hw)
        cx2 = min(patch_w, center_x_in_patch + ex_hw)
        cy1 = max(0, center_y_in_patch - ex_top)
        cy2 = min(patch_h, center_y_in_patch + ex_bot)
        if cx2 > cx1 and cy2 > cy1:
            mask[cy1:cy2, cx1:cx2] = 0

        patch = gray_image[y1:y2, x1:x2]
        pts = cv2.goodFeaturesToTrack(
            patch,
            maxCorners=60,
            qualityLevel=0.012,
            minDistance=3,
            mask=mask,
        )
        if pts is not None and len(pts) > 0:
            pts[:, :, 0] += x1
            pts[:, :, 1] += y1
            self.features = pts.astype(np.float32)
            self.initial_feature_count = len(pts)
        else:
            self.features = None
            self.initial_feature_count = 0

    def track_optical_flow(
        self,
        prev_gray: np.ndarray,
        curr_gray: np.ndarray,
        min_surviving_ratio: float = 0.05,
        min_pts: int = 2,
        max_accel: float = 40.0,
    ) -> tuple[bool, float, float]:
        """Track head movement using Lucas-Kanade optical flow on perimeter features with mathematical verification.
        
        Condition for HOLDING transition:
        Trạng thái = HOLDING <=>
        1. SCRFD(Detect) = False
        2. Số chấm Optical Flow còn sống >= Threshold (>= 5% số chấm gốc) và >= min_pts
        3. Vận tốc di chuyển của cụm chấm ≈ Vận tốc trước đó (độ lệch gia tốc delta_v <= ngưỡng cho phép)
        
        Returns:
            (is_qualified, surviving_ratio, delta_velocity)
        """
        if self.features is None or len(self.features) < min_pts or self.initial_feature_count == 0:
            return False, 0.0, 999.0

        try:
            new_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                prev_gray,
                curr_gray,
                self.features,
                None,
                winSize=(19, 19),
                maxLevel=2,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 15, 0.03),
            )
            valid = (status.flatten() == 1)
            valid_prev = self.features[valid]
            valid_new = new_pts[valid]

            if len(valid_new) < min_pts:
                return False, float(len(valid_new) / max(1, self.initial_feature_count)), 999.0

            deltas = valid_new - valid_prev
            dxs = deltas[:, 0, 0]
            dys = deltas[:, 0, 1]

            # RANSAC/Consensus filter: Remove any rogue points that don't move with the head
            med_dx = float(np.median(dxs))
            med_dy = float(np.median(dys))
            residuals = np.hypot(dxs - med_dx, dys - med_dy)
            inliers = (residuals < 4.5)
            surviving_count = int(np.sum(inliers))

            if surviving_count < min_pts:
                return False, float(surviving_count / max(1, self.initial_feature_count)), 999.0

            dx = float(np.median(dxs[inliers]))
            dy = float(np.median(dys[inliers]))
            valid_pts_to_keep = valid_new[inliers]

            # =========================================================================
            # BỘ LỌC 1: KIỂM TRA "TỶ LỆ CHẤM DI ĐỘNG" (MOTION VECTOR THRESHOLD)
            # Chống lỗi găm khung ma do chấm dính vào vật thể chết (tường, ghế, nẹp cửa):
            # Khi người thật che mặt hoặc ngồi im, cơ thể luôn có độ rung lắc vi mô tự nhiên (> 0.1 px).
            # Ngược lại, nếu 100% số chấm đều có vận tốc bằng 0 (hoặc sai số pixel < 0.1),
            # chứng tỏ người đã di chuyển đi chỗ khác và các chấm bị tuột lại bám trên vật thể chết!
            # Hành động: XÓA NGAY LẬP TỨC, không cho phép giữ khung HOLDING.
            # =========================================================================
            inlier_deltas = deltas[inliers]
            point_speeds = np.hypot(inlier_deltas[:, 0, 0], inlier_deltas[:, 0, 1])
            motion_threshold = 0.10  # Sai số pixel chuẩn cho vật thể tĩnh tuyệt đối
            moving_points = point_speeds >= motion_threshold
            moving_count = int(np.sum(moving_points))

            if moving_count == 0 or np.all(point_speeds < motion_threshold) or float(np.mean(point_speeds)) < 0.08:
                # 100% số chấm đều đứng im bất động trên vật thể chết (tường/ghế) -> Từ chối HOLDING ngay lập tức!
                return False, 0.0, 999.0

            # =========================================================================
            # BỘ LỌC 2: KIỂM TRA MẬT ĐỘ PHÂN TÁN VÀ ĐỘ MỞ CỦA CỤM CHẤM (SPATIAL DISPERSION)
            # =========================================================================
            pts_x = valid_pts_to_keep[:, 0, 0]
            pts_y = valid_pts_to_keep[:, 0, 1]
            span_x = float(np.max(pts_x) - np.min(pts_x))
            span_y = float(np.max(pts_y) - np.min(pts_y))
            ref_w = getattr(self, "base_w", self.w)
            ref_h = getattr(self, "base_h", self.h)

            # (a) Chống co cụm cực đoan (Cluster Collapse): các chấm dính chụm vào 1 điểm/đinh/đốm tường
            if span_x < ref_w * 0.15 and span_y < ref_h * 0.15:
                return False, 0.0, 999.0

            # (b) Chống phân tán vô tổ chức (Cluster Explosion): các chấm bị xé toạc khắp khung hình
            if span_x > ref_w * 2.2 or span_y > ref_h * 2.2:
                return False, 0.0, 999.0

            # Condition 2: Surviving ratio >= Threshold (5% of original anchor points)
            surviving_ratio = float(surviving_count / max(1, self.initial_feature_count))
            ratio_passed = (surviving_ratio >= min_surviving_ratio and surviving_count >= min_pts)

            # Condition 3: Velocity of cluster is consistent with previous head velocity (High-speed adaptive corridor)
            vx_prev, vy_prev = self.center_vel
            delta_v = float(np.hypot(dx - vx_prev, dy - vy_prev))
            allowed_delta_v = max(max_accel, max(abs(vx_prev), abs(vy_prev)) * 1.25 + 12.0)
            velocity_passed = (delta_v <= allowed_delta_v)

            if not (ratio_passed and velocity_passed):
                # Erratic motion or perimeter points dropped below 5% -> Reject holding!
                return False, surviving_ratio, delta_v

            # Dynamic high-speed jump clamp (allows natural ultra-fast motion up to 140px or 1.8x head size)
            max_jump = max(140.0, max(self.w, self.h) * 1.8)
            dx = max(-max_jump, min(max_jump, dx))
            dy = max(-max_jump, min(max_jump, dy))

            # Optical Flow Deadband: Chống vi rung lắc từ nhiễu cảm biến khi ở trạng thái HOLDING
            flow_speed = float(np.hypot(dx, dy))
            if flow_speed < 0.35:
                dx = 0.0
                dy = 0.0

            # Shift center position with actual head movement
            self.cx += dx
            self.cy += dy
            if self.kps is not None:
                self.kps = self.kps.copy()
                self.kps[:, 0] += dx
                self.kps[:, 1] += dy
            if getattr(self, "state", "") == "HOLDING" and getattr(self, "base_w", None) is not None:
                self.w = self.base_w
                self.h = self.base_h
            hw = self.w * 0.5
            hh = self.h * 0.5
            self.bbox = [self.cx - hw, self.cy - hh, self.cx + hw, self.cy + hh]

            # Update velocity smoothly
            self.center_vel = [
                0.35 * dx + 0.65 * vx_prev,
                0.35 * dy + 0.65 * vy_prev,
            ]

            # Update feature positions for continuous tracking
            self.features = valid_pts_to_keep.reshape(-1, 1, 2).astype(np.float32)

            # Re-seed on head if features fall below 15 but still above threshold
            if len(valid_pts_to_keep) < 15 and curr_gray is not None:
                self.refresh_features(curr_gray)

            return True, surviving_ratio, delta_v
        except Exception:
            pass

        return False, 0.0, 999.0

    def update(
        self,
        bbox: list[float],
        kps: np.ndarray | None,
        score: float,
        gray_image: np.ndarray | None = None,
        alpha_pos: float = 0.55,
        alpha_vel: float = 0.35,
    ) -> None:
        """Update track with a newly matched detection."""
        curr_b = [float(x) for x in bbox]
        curr_w = max(24.0, curr_b[2] - curr_b[0])
        curr_h = max(24.0, curr_b[3] - curr_b[1])
        curr_cx = (curr_b[0] + curr_b[2]) * 0.5
        curr_cy = (curr_b[1] + curr_b[3]) * 0.5

        # Check facial completeness (anti-shrink & occlusion detection)
        is_complete = self.check_facial_completeness(curr_b, kps)
        is_known = (self.identity.get("name") not in ("Unknown", None)) and (self.identity.get("id") is not None)

        # Ultra-fast motion support: allows high sprint velocity (up to 280px or 3.2x face width)
        # Face occluded: allows fast head motion while suppressing micro hand jitter
        max_v = max(280.0, curr_w * 3.2) if is_complete else max(50.0, curr_w * 0.8)
        inst_vx = max(-max_v, min(max_v, curr_cx - self.cx))
        inst_vy = max(-max_v, min(max_v, curr_cy - self.cy))

        # Velocity Deadband & Instantaneous Acceleration Update (Chống trôi dạt khi đứng yên)
        v_mag = float(np.hypot(inst_vx, inst_vy))
        if v_mag < 0.60:
            # Triệt tiêu nhiễu khi ngồi tĩnh: vận tốc triệt tiêu dần về 0.0 để khóa cứng khung
            inst_vx = 0.0
            inst_vy = 0.0
            self.center_vel[0] *= 0.5
            self.center_vel[1] *= 0.5
            self.accel[0] *= 0.5
            self.accel[1] *= 0.5
            if float(np.hypot(self.center_vel[0], self.center_vel[1])) < 0.25:
                self.center_vel = [0.0, 0.0]
                self.accel = [0.0, 0.0]
        else:
            # Second-order instantaneous acceleration update
            inst_ax = inst_vx - self.center_vel[0]
            inst_ay = inst_vy - self.center_vel[1]
            self.accel = [
                0.30 * inst_ax + 0.70 * self.accel[0],
                0.30 * inst_ay + 0.70 * self.accel[1],
            ]
            self.center_vel = [
                alpha_vel * inst_vx + (1.0 - alpha_vel) * self.center_vel[0],
                alpha_vel * inst_vy + (1.0 - alpha_vel) * self.center_vel[1],
            ]

        # Second-order scale velocity (approaching or receding from camera)
        inst_vw = curr_w - self.w
        inst_vh = curr_h - self.h
        if float(np.hypot(inst_vw, inst_vh)) < 0.60:
            inst_vw = 0.0
            inst_vh = 0.0
            self.scale_vel = [0.0, 0.0]
        else:
            self.scale_vel = [
                0.25 * inst_vw + 0.75 * self.scale_vel[0],
                0.25 * inst_vh + 0.75 * self.scale_vel[1],
            ]

        # Predictive yaw velocity for best-shot ArcFace trigger
        if kps is not None:
            curr_yaw = self.get_pose_yaw_ratio(kps)
            self.yaw_vel = 0.5 * (curr_yaw - self.last_yaw) + 0.5 * self.yaw_vel
            self.last_yaw = curr_yaw

        if is_known and not is_complete:
            # KHUÔN MẶT BỊ CHE KHUẤT / THIẾU CHI TIẾT (HOLDING):
            # 1. Khóa cứng 100% kích thước base_w, base_h gốc của cả đầu
            self.w = getattr(self, "base_w", self.w)
            self.h = getattr(self, "base_h", self.h)

            # 2. ƯU TIÊN SỐ 1: Bám theo chuyển động thực tế bằng Optical Flow trên viền đầu/tóc
            flow_passed = False
            if gray_image is not None and self.prev_gray is not None and self.features is not None:
                flow_passed, _, _ = self.track_optical_flow(self.prev_gray, gray_image)

            if not flow_passed:
                # 3. Neo đỉnh đầu: Chỉ dịch chuyển khi vị trí đỉnh đầu thay đổi thực sự (> 2.5px)
                top_head = curr_cy - curr_h * 0.5
                target_cy = top_head + self.h * 0.5
                target_cx = curr_cx

                dist_hold = float(np.hypot(target_cx - self.cx, target_cy - self.cy))
                if dist_hold > 2.5:
                    self.cx += 0.15 * (target_cx - self.cx)
                    self.cy += 0.15 * (target_cy - self.cy)

            self.state = "HOLDING"
            self.was_coasting = False  # KHÔNG ép was_coasting = True khi đang bám mặt ổn định
            self.need_recognition = False
        else:
            # Khuôn mặt toàn vẹn, đầy đủ chi tiết:
            # Dual-Zone Adaptive Smoothing:
            # - Vi rung động (< 0.75px) do sai số lượng tử hóa SCRFD: Giữ nguyên vị trí tuyệt đối
            # - Chuyển động vi mô (< 2.5px): Lọc nhiễu mạnh (alpha = 0.22) triệt tiêu 78% rung giật
            # - Chuyển động thật (>= 2.5px): Bắt kịp tức thì (alpha = 0.70) không trễ 1 khung hình
            dist_c = float(np.hypot(curr_cx - self.cx, curr_cy - self.cy))
            if dist_c < 0.75:
                pass
            elif dist_c < 2.5:
                self.cx += 0.22 * (curr_cx - self.cx)
                self.cy += 0.22 * (curr_cy - self.cy)
            else:
                self.cx += 0.70 * (curr_cx - self.cx)
                self.cy += 0.70 * (curr_cy - self.cy)

            # Khử rung kích thước w, h (Scale Stabilization)
            dw = curr_w - self.w
            dh = curr_h - self.h
            if abs(dw) > 1.2:
                self.w += 0.15 * dw
            if abs(dh) > 1.2:
                self.h += 0.15 * dh

            if is_known:
                self.base_w = max(self.base_w, self.w)
                self.base_h = max(self.base_h, self.h)
                if self.was_coasting:
                    # Người này vừa bỏ tay che ra hoặc xoay mặt lại trực diện:
                    # Kích hoạt tái xác thực ArcFace lập tức!
                    self.need_recognition = True
                    self.was_coasting = False
                # Giữ nguyên state nếu đang là HOLDING, chỉ chuyển VERIFIED khi đủ điều kiện
                if getattr(self, "was_verified", False) and getattr(self, "state", "") != "HOLDING":
                    self.state = "VERIFIED"

        hw = self.w * 0.5
        hh = self.h * 0.5
        self.bbox = [self.cx - hw, self.cy - hh, self.cx + hw, self.cy + hh]

        self.kps = kps
        self.score = score
        self.time_since_update = 0
        self.hits += 1
        self.age += 1
        self.frames_since_recognition += 1

        # Refresh Optical Flow feature points on hair & head
        if gray_image is not None:
            self.refresh_features(gray_image)

        # Track motion stability (stillness)
        movement = abs(inst_vx) + abs(inst_vy)
        if movement < 3.0:
            self.still_frames += 1
        else:
            self.still_frames = 0


class FaceStreamTracker:
    """Manages multi-face tracking and selective ArcFace invocation for a video stream."""

    def __init__(
        self,
        re_id_interval: int = 45,  # Re-run ArcFace every ~45 frames (approx 1.5s at 30 FPS)
        max_lost_frames: int = 10,  # Grace period before deleting lost track (~300ms)
        max_coasting_frames: int = 45,  # Hold state & trajectory coasting when face turns away (~1.5s)
        iou_threshold: float = 0.30,  # Minimum IoU to match tracks
        ambiguity_distance: float = 80.0,  # If two faces come within 80px, re-verify both
        detection_interval: int = 2,  # Interleaved predictive detection interval (1=every frame, 2=every 2nd frame when stable)
    ) -> None:
        self.re_id_interval = re_id_interval
        self.max_lost_frames = max_lost_frames
        self.max_coasting_frames = max_coasting_frames
        self.iou_threshold = iou_threshold
        self.ambiguity_distance = ambiguity_distance
        self.detection_interval = detection_interval

        self.tracks: list[FaceTrack] = []
        self._next_id = 1
        self.prev_gray: np.ndarray | None = None
        self._prev_gray_buf: np.ndarray | None = None
        self.frame_count: int = 0
        self.force_detect_next: bool = True

    def _save_prev_gray(self, gray_image: np.ndarray | None) -> None:
        """Zero-allocation reusable buffer copy for optical flow reference."""
        if gray_image is not None and gray_image.size > 0:
            if self._prev_gray_buf is None or self._prev_gray_buf.shape != gray_image.shape:
                self._prev_gray_buf = np.empty_like(gray_image)
            np.copyto(self._prev_gray_buf, gray_image)
            self.prev_gray = self._prev_gray_buf

    def should_run_detection(self, img_w: int, img_h: int) -> bool:
        """Determines if heavy SCRFD deep detection must run or if predictive tracking is sufficient."""
        # 1. Startup or forced trigger
        if self.force_detect_next or len(self.tracks) == 0:
            self.force_detect_next = False
            return True

        # 2. Check if any track needs recognition or is in an invalid state or lost
        for t in self.tracks:
            # Đồng bộ triệt để cho cả 3 trạng thái: VERIFIED, HOLDING, UNKNOWN
            is_valid_state = (t.state in ("VERIFIED", "HOLDING", "UNKNOWN")) or getattr(t, "was_verified", False) or getattr(t, "is_stranger", False)
            if t.need_recognition or not is_valid_state or t.time_since_update > 0:
                return True
            # Any track near camera border might have a new person entering nearby
            if t.is_touching_edge(img_w, img_h, threshold=35.0):
                return True
            # High acceleration or sudden jerk
            accel_mag = float(np.hypot(t.accel[0], t.accel[1]))
            if accel_mag > 14.0:
                return True

        # 3. Interleaved detection interval
        if self.frame_count % self.detection_interval == 0:
            return True

        return False

    def predictive_step(
        self,
        gray_image: np.ndarray | None = None,
        image_shape: tuple[int, int] | None = None,
    ) -> list[tuple[FaceTrack, bool]]:
        """Ultra-fast zero-detection predictive tracking step (~0.1ms).
        Advances active verified tracks using optical flow and second-order kinematics.
        """
        self.frame_count += 1
        img_h, img_w = image_shape if image_shape is not None else (720, 1280)

        for t in self.tracks:
            t.age += 1
            t.frames_since_recognition += 1

            flow_passed = False
            if self.prev_gray is not None and gray_image is not None and t.features is not None:
                flow_passed, _, _ = t.track_optical_flow(self.prev_gray, gray_image)

            if not flow_passed:
                # Apply second-order kinematic trajectory only if velocity exceeds deadband
                v_mag = float(np.hypot(t.center_vel[0], t.center_vel[1]))
                if v_mag > 0.25:
                    kin_dx = t.center_vel[0] + 0.5 * t.accel[0]
                    kin_dy = t.center_vel[1] + 0.5 * t.accel[1]
                    t.cx += kin_dx
                    t.cy += kin_dy
                    if abs(t.scale_vel[0]) > 0.3:
                        t.w = max(24.0, t.w + t.scale_vel[0])
                    if abs(t.scale_vel[1]) > 0.3:
                        t.h = max(24.0, t.h + t.scale_vel[1])
                    if t.kps is not None:
                        t.kps = t.kps.copy()
                        t.kps[:, 0] += kin_dx
                        t.kps[:, 1] += kin_dy

            hw = t.w * 0.5
            hh = t.h * 0.5
            t.bbox = [t.cx - hw, t.cy - hh, t.cx + hw, t.cy + hh]

            # If track touches edge, trigger full detection next frame
            if t.is_touching_edge(img_w, img_h, threshold=20.0):
                self.force_detect_next = True

        self._save_prev_gray(gray_image)

        # Return visible tracks (need_recognition is False in predictive step)
        visible_results: list[tuple[FaceTrack, bool]] = []
        for t in self.tracks:
            if not t.is_out_of_bounds(img_w, img_h):
                visible_results.append((t, False))
        return visible_results

    @staticmethod
    def calc_iou(b1: list[float], b2: list[float]) -> float:
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
        return inter / union if union > 0.0 else 0.0

    @staticmethod
    def calc_center_dist(b1: list[float], b2: list[float]) -> float:
        c1x = (b1[0] + b1[2]) * 0.5
        c1y = (b1[1] + b1[3]) * 0.5
        c2x = (b2[0] + b2[2]) * 0.5
        c2y = (b2[1] + b2[3]) * 0.5
        return float(np.hypot(c1x - c2x, c1y - c2y))

    def update_tracks(
        self,
        det_bboxes: np.ndarray,
        det_kpss: np.ndarray | None,
        gray_image: np.ndarray | None = None,
        image_shape: tuple[int, int] | None = None,
    ) -> list[tuple[FaceTrack, bool]]:
        """Update tracker with SCRFD detections and Lucas-Kanade Optical Flow head tracking."""
        self.frame_count += 1
        N = det_bboxes.shape[0] if det_bboxes is not None else 0
        detections = []
        for i in range(N):
            b = det_bboxes[i][:4].tolist()
            s = float(det_bboxes[i][4]) if det_bboxes.shape[1] > 4 else 1.0
            k = det_kpss[i] if det_kpss is not None and i < len(det_kpss) else None

            # BỘ LỌC CHỐNG FALSE POSITIVE TRÊN NẾP ÁO, CỔ ÁO VÀ VÁCH TƯỜNG:
            # 1. Điểm tin cậy tối thiểu SCRFD: s >= 0.50
            if s < 0.50:
                continue

            # 2. Kích thước tối thiểu hộp: bỏ qua các nhiễu quá nhỏ < 20x20 px
            bw = b[2] - b[0]
            bh = b[3] - b[1]
            if bw < 20.0 or bh < 20.0:
                continue

            # 3. Tỷ lệ khung hình (Aspect Ratio): Khuôn mặt người bình thường có W/H nằm trong [0.50, 1.55]
            # Loại bỏ các vết dọc hẹp như khe cửa/mép tường (W/H < 0.50) hoặc thanh ngang dài (W/H > 1.55)
            aspect = bw / max(1.0, bh)
            if aspect < 0.50 or aspect > 1.55:
                continue

            # 4. Kiểm tra mốc hình học (nếu có đủ 5 điểm landmarks):
            # [left_eye, right_eye, nose, left_mouth, right_mouth]
            if k is not None and len(k) >= 5:
                eye_span = abs(float(k[1][0] - k[0][0]))
                eye_y = float(k[0][1] + k[1][1]) * 0.5
                mouth_y = float(k[3][1] + k[4][1]) * 0.5
                # Khoảng cách 2 mắt không được quá hẹp so với chiều rộng hộp
                if eye_span < bw * 0.12:
                    continue
                # Miệng phải nằm dưới mắt
                if mouth_y < eye_y - 2.0:
                    continue

            detections.append((b, k, s))

        # Predict active track positions
        predictions = [t.predict() for t in self.tracks]

        matched_tracks: set[int] = set()
        matched_dets: set[int] = set()

        # Step 1: Global Maximum-IoU Assignment (Highest IoU matched first to prevent identity hijacking)
        iou_candidates: list[tuple[float, int, int]] = []
        for di, (db, dk, ds) in enumerate(detections):
            for ti, pred_b in enumerate(predictions):
                score = self.calc_iou(pred_b, db)
                if score >= self.iou_threshold:
                    iou_candidates.append((score, di, ti))

        iou_candidates.sort(key=lambda x: x[0], reverse=True)

        for score, di, ti in iou_candidates:
            if di not in matched_dets and ti not in matched_tracks:
                matched_dets.add(di)
                matched_tracks.add(ti)
                t = self.tracks[ti]
                db, dk, ds = detections[di]
                t.update(db, dk, ds, gray_image=gray_image)

                # Periodic re-identification with Adaptive Motion
                # Đồng bộ cho cả 3 trạng thái: VERIFIED, HOLDING, UNKNOWN
                is_valid_state = (t.state in ("VERIFIED", "HOLDING", "UNKNOWN")) or getattr(t, "was_verified", False) or getattr(t, "is_stranger", False) or getattr(t, "auto_registered", False) or (t.identity.get("id") is not None and t.identity.get("name") not in ("Unknown", None))
                if is_valid_state:
                    is_known = (t.identity.get("name") not in ("Unknown", None)) and (t.identity.get("id") is not None)
                    effective_interval = (self.re_id_interval * 2) if (is_known and t.still_frames > 15) else self.re_id_interval
                    if t.frames_since_recognition >= effective_interval:
                        t.need_recognition = True

        # Fallback: Multi-scale Proximity Association for fast-moving faces
        unmatched_det_indices = [di for di in range(len(detections)) if di not in matched_dets]
        for di in unmatched_det_indices:
            db, dk, ds = detections[di]
            curr_w = db[2] - db[0]
            curr_h = db[3] - db[1]
            best_dist = 999999.0
            best_ti = -1
            for ti, pred_b in enumerate(predictions):
                if ti in matched_tracks:
                    continue
                t = self.tracks[ti]
                # Scale check: allows fast approach / retreat from camera
                scale_ratio = curr_w / max(1.0, t.w)
                if scale_ratio < 0.40 or scale_ratio > 2.20:
                    continue

                # Multi-point distance: Check distance against predicted position and last known position
                dist_pred = self.calc_center_dist(pred_b, db)
                dist_last = self.calc_center_dist(t.bbox, db)
                dist = min(dist_pred, dist_last)

                # Dynamic Directional Kinematic Gate (Hành lang chuyển động theo hướng và vận tốc):
                det_cx = (db[0] + db[2]) * 0.5
                det_cy = (db[1] + db[3]) * 0.5
                disp_x = det_cx - t.cx
                disp_y = det_cy - t.cy
                disp_len = float(np.hypot(disp_x, disp_y))

                vel_len = float(np.hypot(t.center_vel[0], t.center_vel[1]))
                is_aligned = False
                if vel_len > 4.0 and disp_len > 8.0:
                    cos_theta = (t.center_vel[0] * disp_x + t.center_vel[1] * disp_y) / (vel_len * disp_len)
                    is_aligned = (cos_theta > 0.40) # Motion vector aligned along path

                # If moving in aligned direction, expand corridor distance threshold up to 350px!
                if is_aligned:
                    max_allowed = max(350.0, vel_len * 2.5 + max(t.w, t.h) * 3.5)
                else:
                    max_allowed = max(max(t.w, t.h) * 2.8, 180.0)

                if dist < max_allowed and dist < best_dist:
                    best_dist = dist
                    best_ti = ti

            if best_ti >= 0:
                matched_dets.add(di)
                matched_tracks.add(best_ti)
                t = self.tracks[best_ti]
                t.update(db, dk, ds, gray_image=gray_image)
                if t.state == "UNKNOWN" or (t.state == "HOLDING" and t.frames_since_recognition >= self.re_id_interval):
                    t.need_recognition = True

        # Step 2: Spawn new tracks for unmatched detections (Always need ArcFace!)
        for di, (db, dk, ds) in enumerate(detections):
            if di not in matched_dets:
                new_track = FaceTrack(self._next_id, db, dk, ds)
                self._next_id += 1
                new_track.need_recognition = True
                new_track.state = "UNKNOWN"
                if gray_image is not None:
                    new_track.refresh_features(gray_image)
                self.tracks.append(new_track)
                matched_tracks.add(len(self.tracks) - 1)

        # Step 3: Handle unmatched tracks (Tạm giữ / Lucas-Kanade Optical Flow Head Tracking)
        for ti, t in enumerate(self.tracks):
            if ti not in matched_tracks:
                t.time_since_update += 1

                # ĐIỀU KIỆN TOÁN HỌC CHUYỂN TRẠNG THÁI HOLDING:
                # Trạng thái = HOLDING <=>
                # 1. SCRFD(Detect) = False (ti not in matched_tracks)
                # 2. Số chấm Optical Flow còn sống >= 5% số chấm gốc (surviving_ratio >= 0.05)
                # 3. Vận tốc di chuyển của cụm chấm ≈ Vận tốc trước đó (độ lệch gia tốc delta_v <= allowed)
                # 4. BỘ LỌC 1: Tỷ lệ chấm di động > 0 (loại bỏ 100% vật thể chết v < 0.1 px)
                # 5. BỘ LỌC 2: Mật độ phân tán của cụm chấm tương xứng kích thước đầu
                is_known = (t.identity.get("name") not in ("Unknown", None)) and (t.identity.get("id") is not None)

                flow_passed = False
                surviving_ratio = 0.0
                delta_v = 999.0

                if self.prev_gray is not None and gray_image is not None and is_known:
                    flow_passed, surviving_ratio, delta_v = t.track_optical_flow(self.prev_gray, gray_image)

                if flow_passed and is_known:
                    # THỎA MÃN ĐẦY ĐỦ ĐIỀU KIỆN TOÁN HỌC:
                    # Các chấm viền ngoài vẫn di chuyển nhịp nhàng theo đầu -> Cho phép găm khung vàng HOLDING
                    t.state = "HOLDING"
                    t.was_coasting = True
                else:
                    # KHÔNG THỎA MÃN ĐIỀU KIỆN (Mất chấm viền ngoài, vận tốc bất thường, hoặc dính vào vật thể chết):
                    # NGƯỜI ĐÃ BIẾN MẤT HOẶC DI CHUYỂN RA NGOÀI TẦM NHÌN:
                    # Tuyệt đối KHÔNG gán HOLDING mù quáng! Chuyển ngay sang UNKNOWN, xóa vận tốc và tiêu hủy tức thì!
                    t.state = "UNKNOWN"
                    t.was_coasting = False
                    t.center_vel[0] = 0.0
                    t.center_vel[1] = 0.0
                    t.time_since_update = max(t.time_since_update, 2)

        # Step 4: Proactive Identity Conflict & Body Trail Suppression
        live_names: set[str] = set()
        live_ids: set[str] = set()
        live_faces: list[tuple[float, float, float]] = []
        for t in self.tracks:
            if t.time_since_update == 0:
                name = t.identity.get("name")
                uid = t.identity.get("id")
                if name and name != "Unknown":
                    live_names.add(name)
                if uid:
                    live_ids.add(uid)
                live_faces.append((t.cx, t.cy, max(t.w, t.h)))

        visible_tracks_check = [
            t for t in self.tracks
            if t.time_since_update == 0 or t.state in ("HOLDING", "COASTING")
        ]
        for i in range(len(visible_tracks_check)):
            for j in range(i + 1, len(visible_tracks_check)):
                t_i = visible_tracks_check[i]
                t_j = visible_tracks_check[j]
                d = self.calc_center_dist(t_i.bbox, t_j.bbox)
                avg_size = (max(t_i.w, t_i.h) + max(t_j.w, t_j.h)) * 0.5
                collision_dist = min(self.ambiguity_distance, max(30.0, avg_size * 0.85))
                if d < collision_dist:
                    t_i.need_recognition = True
                    t_j.need_recognition = True
                    if t_i.time_since_update == 0 and t_j.state in ("HOLDING", "COASTING"):
                        t_j.state = "UNKNOWN"
                        t_j.need_recognition = True
                    elif t_j.time_since_update == 0 and t_i.state in ("HOLDING", "COASTING"):
                        t_i.state = "UNKNOWN"
                        t_i.need_recognition = True

        # Step 5: Prune dead tracks & eradicate boundary/body ghost frames
        if image_shape is not None:
            img_h, img_w = image_shape
        elif gray_image is not None and gray_image.size > 0:
            img_h, img_w = gray_image.shape[:2]
        else:
            img_h, img_w = (240, 320)

        alive_tracks: list[FaceTrack] = []
        for t in self.tracks:
            # 1. Edge boundary exit
            is_at_border = t.is_touching_edge(img_w, img_h, threshold=6.0)
            is_off_screen = t.is_out_of_bounds(img_w, img_h)
            if (is_at_border or is_off_screen) and t.time_since_update >= 2:
                continue

            # 2. Tentative track pruning (false positive noise)
            if t.hits < 2 and t.time_since_update >= 2:
                continue

            # 3. GHOST SUPPRESSION (Industrial Standard):
            # A person cannot be in two places at once!
            if t.time_since_update > 0:
                t_name = t.identity.get("name")
                t_id = t.identity.get("id")
                # (a) If this person's identity is ALREADY actively detected elsewhere -> kill ghost instantly!
                if (t_name and t_name != "Unknown" and t_name in live_names) or (t_id and t_id in live_ids):
                    continue

                # (b) If holding box is located on the chest/body of an active person (< 1.6 * size) -> kill ghost!
                is_on_body = False
                for lcx, lcy, lsize in live_faces:
                    if float(np.hypot(t.cx - lcx, t.cy - lcy)) < lsize * 1.6:
                        is_on_body = True
                        break
                if is_on_body:
                    continue

                # (c) Motionless empty holding timeout:
                # If optical flow has head points -> allow holding up to 45 frames (~1.5s).
                # If optical flow has NO head points (empty wall/clothes) -> EXPIRE IN 3 FRAMES (0.1s - 0.3s max)!
                has_head_flow = (t.features is not None and len(t.features) >= 2 and getattr(t, "was_coasting", False))
                max_hold = 45 if has_head_flow else 3
                if t.time_since_update > max_hold:
                    continue

            if t.state not in ("HOLDING", "COASTING") and t.time_since_update >= 2:
                continue

            alive_tracks.append(t)

        # Track-level Non-Maximum Suppression (Eliminate duplicate tracks while allowing 2 different persons to overlap)
        nms_tracks: list[FaceTrack] = []
        sorted_tracks = sorted(
            alive_tracks,
            key=lambda trk: (
                2 if trk.state == "VERIFIED" else (1 if trk.state in ("HOLDING", "COASTING") else 0),
                trk.hits,
                -trk.time_since_update
            ),
            reverse=True
        )
        for cand in sorted_tracks:
            overlap = False
            for kept in nms_tracks:
                iou = self.calc_iou(cand.bbox, kept.bbox)
                if iou <= 0.0:
                    continue

                # Cơ chế chuẩn công nghiệp: Nếu xác định là 2 người khác nhau thì cho phép đè lên khung hình của nhau!
                cand_name = cand.identity.get("name") if isinstance(cand.identity, dict) else None
                kept_name = kept.identity.get("name") if isinstance(kept.identity, dict) else None
                cand_id = cand.identity.get("id") if isinstance(cand.identity, dict) else None
                kept_id = kept.identity.get("id") if isinstance(kept.identity, dict) else None

                cand_has_id = bool((cand_name and cand_name != "Unknown") or cand_id is not None)
                kept_has_id = bool((kept_name and kept_name != "Unknown") or kept_id is not None)

                is_different_persons = False
                if cand_has_id and kept_has_id:
                    # Hai người đã có danh tính khác biệt -> Cho phép đè lên khung hình nhau
                    if (cand_name != kept_name) or (cand_id != kept_id):
                        is_different_persons = True
                elif cand.time_since_update == 0 and kept.time_since_update == 0 and cand.track_id != kept.track_id:
                    # Cả 2 đều là khuôn mặt thực tế được SCRFD phát hiện trong khung hình hiện tại
                    # Cho phép đè lên nhau (ví dụ đứng sát nhau, ôm nhau, người trước người sau), trừ khi trùng khít > 0.88 (lỗi duplicate box)
                    if iou < 0.88:
                        is_different_persons = True

                if is_different_persons:
                    continue

                # Chỉ triệt tiêu khi cùng một người bị phân mảnh thành 2 khung hoặc khung ghost đè lên
                if iou > 0.55:
                    overlap = True
                    break

            if not overlap:
                nms_tracks.append(cand)

        self.tracks = nms_tracks

        # Save current gray frame for next frame's optical flow calculation
        self._save_prev_gray(gray_image)

        # Return visible tracks in current frame:
        visible_results: list[tuple[FaceTrack, bool]] = []
        for t in self.tracks:
            is_at_border = t.is_touching_edge(img_w, img_h, threshold=6.0)
            is_off_screen = t.is_out_of_bounds(img_w, img_h)
            if (is_at_border or is_off_screen) and t.time_since_update >= 2:
                continue

            if t.time_since_update == 0:
                visible_results.append((t, t.need_recognition))
            elif t.state in ("HOLDING", "COASTING"):
                has_head_flow = (t.features is not None and len(t.features) >= 2 and getattr(t, "was_coasting", False))
                max_hold = 45 if has_head_flow else 3
                if t.time_since_update <= max_hold:
                    visible_results.append((t, False))
        return visible_results

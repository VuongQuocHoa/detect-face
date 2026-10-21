"""Image decoding and validation helpers."""

import cv2
import numpy as np
from fastapi import HTTPException, UploadFile


async def read_upload(file: UploadFile) -> bytes:
    """Read and validate an uploaded image payload."""
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded image is empty.")
    return content


def decode_image(image_bytes: bytes) -> np.ndarray:
    """Decode image bytes into a BGR OpenCV image."""
    array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise ValueError("Invalid or unsupported image format.")
    return image


def crop_and_save_face(
    image: np.ndarray,
    bbox: list[float],
    user_id: str,
    margin_ratio: float = 0.25,
) -> str | None:
    """Crop face region with margin, scale to crisp avatar size, and save to FACES_DIR."""
    try:
        from pathlib import Path
        from app.config import FACES_DIR

        h, w = image.shape[:2]
        x1, y1, x2, y2 = bbox
        bw = x2 - x1
        bh = y2 - y1
        pad_x = bw * margin_ratio
        pad_y = bh * margin_ratio

        cx1 = max(0, int(x1 - pad_x))
        cy1 = max(0, int(y1 - pad_y))
        cx2 = min(w, int(x2 + pad_x))
        cy2 = min(h, int(y2 + pad_y))

        crop = image[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            crop = image[max(0, int(y1)):min(h, int(y2)), max(0, int(x1)):min(w, int(x2))]
        if crop.size == 0:
            return None

        ch, cw = crop.shape[:2]
        if max(ch, cw) > 320:
            scale = 320 / max(ch, cw)
            crop = cv2.resize(
                crop,
                (int(cw * scale), int(ch * scale)),
                interpolation=cv2.INTER_AREA,
            )

        out_path = Path(FACES_DIR) / f"{user_id}.jpg"
        cv2.imwrite(str(out_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
        return str(out_path)
    except Exception as exc:
        print(f"Failed to crop and save face for {user_id}: {exc}")
        return None


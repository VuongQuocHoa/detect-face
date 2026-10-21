"""Ultra-high performance binary protocol for real-time face stream.

Replaces slow JSON serialization with fixed-width memory-packed structs.
Triệt tiêu 100% rác bộ nhớ (Garbage Collection) và độ trễ phân tích chuỗi.

Packet Structure:
Header (8 bytes):
  - [0..1] Magic: 0xBE, 0xEF
  - [2..5] inference_ms: float32 (little endian)
  - [6..7] face_count: uint16 (little endian)

Each Face Record (31 + N + M bytes):
  - [0..15] bbox: 4 x float32 (x1, y1, x2, y2)
  - [16..19] similarity: float32 (0.0 if None)
  - [20..23] distance: float32 (0.0 if None)
  - [24..27] track_id: int32
  - [28] matched: uint8 (1 = known, 0 = unknown)
  - [29] name_len: uint8 (N)
  - [30 .. 29+N] name: UTF-8 string
  - [30+N] id_len: uint8 (M)
  - [31+N .. 30+N+M] id: UTF-8 string
"""

import struct
from typing import Any

MAGIC = b"\xbe\xef"
MAGIC_BYTES = (0xBE, 0xEF)


def pack_recognition_binary(faces: list[dict[str, Any]], inference_ms: float) -> bytes:
    """Pack recognition results into a zero-overhead binary buffer."""
    face_count = len(faces)
    # Header: 2s (magic) + f (inference_ms) + H (face_count) -> 8 bytes
    header = struct.pack("<2sfH", MAGIC, float(inference_ms), face_count)
    buf = bytearray(header)

    for face in faces:
        bbox = face.get("bbox") or [0.0, 0.0, 0.0, 0.0]
        sim = float(face.get("similarity") or 0.0)
        dist = float(face.get("distance") or 0.0)
        track_id = int(face.get("track_id") or 0)
        name = str(face.get("name") or "Unknown")
        user_id = str(face.get("id") or "")

        name_bytes = name.encode("utf-8")[:64]
        id_bytes = user_id.encode("utf-8")[:64]
        state = face.get("state")
        if state in ("COASTING", "HOLDING"):
            matched = 2  # 2 = HOLDING / COASTING (Tạm giữ)
        elif name != "Unknown" and user_id:
            matched = 1  # 1 = VERIFIED
        else:
            matched = 0  # 0 = UNKNOWN

        # Fixed segment: 4f (bbox) + 2f (sim, dist) + i (track_id) + B (matched/state) + B (name_len)
        face_fixed = struct.pack(
            "<ffffffiBB",
            float(bbox[0]),
            float(bbox[1]),
            float(bbox[2]),
            float(bbox[3]),
            sim,
            dist,
            track_id,
            matched,
            len(name_bytes),
        )
        buf.extend(face_fixed)
        buf.extend(name_bytes)
        buf.append(len(id_bytes))
        buf.extend(id_bytes)

    return bytes(buf)


def unpack_recognition_binary(data: bytes) -> dict[str, Any]:
    """Unpack binary buffer back into Python dict (useful for test scripts & benchmarks)."""
    if len(data) < 8 or data[0] != 0xBE or data[1] != 0xEF:
        raise ValueError("Invalid binary magic header (expected 0xBEEF)")

    _, inference_ms, face_count = struct.unpack("<2sfH", data[:8])
    offset = 8
    faces: list[dict[str, Any]] = []

    for _ in range(face_count):
        if offset + 30 > len(data):
            break
        x1, y1, x2, y2, sim, dist, track_id, matched, name_len = struct.unpack_from(
            "<ffffffiBB", data, offset
        )
        offset += 30

        name_bytes = data[offset : offset + name_len]
        offset += name_len
        name = name_bytes.decode("utf-8", errors="replace")

        if offset >= len(data):
            break
        id_len = data[offset]
        offset += 1

        id_bytes = data[offset : offset + id_len]
        offset += id_len
        user_id = id_bytes.decode("utf-8", errors="replace") if id_len > 0 else None

        if matched == 2:
            state_str = "HOLDING"
            is_matched = True
        elif matched == 1:
            state_str = "VERIFIED"
            is_matched = True
        else:
            state_str = "UNKNOWN"
            is_matched = False

        faces.append({
            "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
            "name": name,
            "id": user_id,
            "similarity": round(sim, 4) if sim > 0 else None,
            "distance": round(dist, 4) if dist > 0 else None,
            "track_id": track_id,
            "matched": is_matched,
            "state": state_str,
        })

    return {
        "faces": faces,
        "inference_ms": round(inference_ms, 1),
    }

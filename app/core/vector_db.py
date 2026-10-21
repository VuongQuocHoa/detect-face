"""ChromaDB persistence and similarity search operations."""

from typing import Any
from uuid import uuid4

import chromadb
from chromadb.api.models.Collection import Collection

from app.config import CHROMA_DB_PATH, FACES_DIR, INSIGHTFACE_MODEL_NAME

import os
import threading
from datetime import datetime
from pathlib import Path
import numpy as np

_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
_collection_name = f"user_faces_{INSIGHTFACE_MODEL_NAME}"
_collection: Collection = _client.get_or_create_collection(
    name=_collection_name,
    metadata={"hnsw:space": "cosine"},
)

# Ultra-Fast Zero-Latency In-Memory Matrix Cache with Thread Safety
_cache_lock = threading.Lock()
_cached_matrix: np.ndarray | None = None
_cached_metas: list[dict[str, Any]] = []


def _refresh_cache() -> None:
    """Pre-compute normalized embedding matrix in RAM for microsecond cosine search."""
    global _cached_matrix, _cached_metas
    try:
        data = _collection.get(include=["embeddings", "metadatas"])
        embeddings = data.get("embeddings")
        metadatas = data.get("metadatas")
        if embeddings is not None and len(embeddings) > 0:
            mat = np.asarray(embeddings, dtype=np.float32)
            norms = np.linalg.norm(mat, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            norm_mat = mat / norms
            m_list = list(metadatas) if metadatas is not None else []
            with _cache_lock:
                _cached_matrix = norm_mat
                _cached_metas = m_list
        else:
            with _cache_lock:
                _cached_matrix = None
                _cached_metas = []
    except Exception as exc:
        print(f"Vector DB cache refresh warning: {exc}")
        with _cache_lock:
            _cached_matrix = None
            _cached_metas = []


_refresh_cache()


def add_face(user_id: str, name: str, embedding: list[float]) -> None:
    """Store one face embedding and its user metadata."""
    if not embedding:
        raise ValueError("Embedding must not be empty.")

    _collection.add(
        ids=[f"{user_id}:{uuid4()}"],
        embeddings=[embedding],
        metadatas=[{"user_id": user_id, "name": name}],
    )
    _refresh_cache()


def search_face(embedding: list[float], threshold: float) -> dict[str, Any]:
    """Return the closest user using ultra-fast in-memory matrix multiplication."""
    if not embedding:
        raise ValueError("Embedding must not be empty.")

    with _cache_lock:
        mat = _cached_matrix
        metas = _cached_metas

    if mat is not None and len(metas) > 0:
        query = np.asarray(embedding, dtype=np.float32).reshape(-1)
        q_norm = np.linalg.norm(query)
        if q_norm > 0:
            query = query / q_norm
        sims = np.dot(mat, query)
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])
        dist = max(0.0, 1.0 - best_sim)

        if best_sim >= threshold and best_idx < len(metas):
            meta = metas[best_idx]
            return {
                "id": str(meta["user_id"]),
                "name": str(meta["name"]),
                "distance": round(dist, 4),
                "similarity": round(best_sim, 4),
                "matched": True,
            }
        elif best_idx < len(metas):
            meta = metas[best_idx]
            return {
                "id": None,
                "name": "Unknown",
                "candidate_id": str(meta["user_id"]),
                "candidate_name": str(meta["name"]),
                "distance": round(dist, 4),
                "similarity": round(best_sim, 4),
                "matched": False,
            }

    return {
        "id": None,
        "name": "Unknown",
        "distance": 1.0,
        "similarity": 0.0,
        "matched": False,
    }


def list_users() -> list[dict[str, Any]]:
    """Return a summary of all registered users and strangers with photo information."""
    data = _collection.get()
    ids = data.get("ids") or []
    metadatas = data.get("metadatas") or []
    users_map: dict[str, dict[str, Any]] = {}
    for doc_id, meta in zip(ids, metadatas):
        if not meta:
            continue
        uid = str(meta.get("user_id", ""))
        name = str(meta.get("name", "Unknown"))
        if not uid:
            continue
        if uid not in users_map:
            photo_file = Path(FACES_DIR) / f"{uid}.jpg"
            has_photo = photo_file.is_file()
            created_str = ""
            if has_photo:
                try:
                    mtime = os.path.getmtime(photo_file)
                    created_str = datetime.fromtimestamp(mtime).strftime("%d/%m/%Y %H:%M:%S")
                except Exception:
                    created_str = ""

            is_stranger = uid.lower().startswith("stranger_") or name.lower().startswith("người lạ")
            users_map[uid] = {
                "user_id": uid,
                "name": name,
                "sample_count": 1,
                "has_photo": has_photo,
                "photo_url": f"/api/v1/users/{uid}/photo" if has_photo else None,
                "is_stranger": is_stranger,
                "created_at": created_str,
            }
        else:
            users_map[uid]["sample_count"] += 1
    return list(users_map.values())


def list_registered_members() -> list[dict[str, Any]]:
    """Return only officially registered members (non-strangers)."""
    return [u for u in list_users() if not u.get("is_stranger")]


def list_strangers() -> list[dict[str, Any]]:
    """Return only auto-registered strangers."""
    return [u for u in list_users() if u.get("is_stranger")]


def delete_user(user_id: str) -> int:
    """Delete all embeddings associated with the given user_id and remove photo."""
    data = _collection.get(where={"user_id": user_id})
    ids = data.get("ids") or []
    if ids:
        _collection.delete(ids=ids)
    _refresh_cache()

    photo_path = Path(FACES_DIR) / f"{user_id}.jpg"
    if photo_path.is_file():
        try:
            photo_path.unlink()
        except Exception as exc:
            print(f"Warning deleting photo for {user_id}: {exc}")

    return len(ids)


def delete_all_strangers() -> int:
    """Delete all auto-registered stranger records and their photos."""
    strangers = list_strangers()
    deleted = 0
    for s in strangers:
        deleted += delete_user(s["user_id"])
    return deleted


def convert_stranger(old_user_id: str, new_user_id: str, new_name: str) -> bool:
    """Convert an auto-registered stranger into an official registered member."""
    data = _collection.get(where={"user_id": old_user_id}, include=["embeddings", "metadatas"])
    ids = data.get("ids")
    embeddings = data.get("embeddings")
    if ids is None or len(ids) == 0 or embeddings is None or len(embeddings) == 0:
        return False

    _collection.delete(ids=ids)

    new_ids = [f"{new_user_id}:{uuid4()}" for _ in range(len(embeddings))]
    new_metas = [{"user_id": new_user_id, "name": new_name} for _ in range(len(embeddings))]
    _collection.add(
        ids=new_ids,
        embeddings=embeddings,
        metadatas=new_metas,
    )
    _refresh_cache()

    old_photo = Path(FACES_DIR) / f"{old_user_id}.jpg"
    new_photo = Path(FACES_DIR) / f"{new_user_id}.jpg"
    if old_photo.is_file():
        try:
            if new_photo.exists():
                new_photo.unlink()
            old_photo.rename(new_photo)
        except Exception as exc:
            print(f"Warning renaming photo {old_user_id} -> {new_user_id}: {exc}")

    return True


def update_user_info(old_user_id: str, new_user_id: str, new_name: str) -> bool:
    """Update name and/or user_id of an existing registered user or stranger."""
    return convert_stranger(old_user_id, new_user_id, new_name)


def delete_all_registered_members() -> int:
    """Delete all registered official members from ChromaDB and disk."""
    members = list_registered_members()
    deleted = 0
    for m in members:
        deleted += delete_user(m["user_id"])
    return deleted


def delete_all_faces() -> int:
    """Delete all face records (both members and strangers) from ChromaDB and disk."""
    all_users = list_users()
    deleted = 0
    for u in all_users:
        deleted += delete_user(u["user_id"])
    return deleted


def get_next_stranger_info() -> tuple[str, str]:
    """Find the next available Stranger ID and Name sequentially."""
    import re
    with _cache_lock:
        metas = list(_cached_metas)
    max_idx = 0
    for meta in metas:
        uid = str(meta.get("user_id", ""))
        m = re.search(r"stranger_(\d+)", uid, re.IGNORECASE)
        if m:
            val = int(m.group(1))
            if val > max_idx:
                max_idx = val
        else:
            name = str(meta.get("name", ""))
            m2 = re.search(r"Người lạ\s*#?(\d+)", name, re.IGNORECASE)
            if m2:
                val = int(m2.group(1))
                if val > max_idx:
                    max_idx = val

    next_idx = max_idx + 1
    new_id = f"stranger_{next_idx:03d}"
    new_name = f"Người lạ #{next_idx}"
    return new_id, new_name



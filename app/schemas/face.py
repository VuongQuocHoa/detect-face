"""Pydantic response schemas for face API endpoints."""

from pydantic import BaseModel, Field


class FaceMatch(BaseModel):
    """Recognition result for one detected face."""

    bbox: list[float] = Field(min_length=4, max_length=4)
    name: str
    id: str | None = None
    distance: float | None = None
    similarity: float | None = None


class RecognitionResponse(BaseModel):
    """Recognition response containing every detected face."""

    faces: list[FaceMatch]


class DetectedFaceCrop(BaseModel):
    """Cropped face preview for candidate selection in registration."""

    index: int
    bbox: list[float] = Field(min_length=4, max_length=4)
    thumbnail: str
    area: float
    is_registered: bool = False
    registered_id: str | None = None
    registered_name: str | None = None
    similarity: float | None = None


class DetectFacesResponse(BaseModel):
    """List of all detected faces extracted from an image."""

    total_faces: int
    faces: list[DetectedFaceCrop]


class RegisterResponse(BaseModel):
    """Registration response."""

    user_id: str
    name: str
    message: str


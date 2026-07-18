"""Pydantic schemas for the AeroSense API."""
from __future__ import annotations
from typing import Literal, Union
from pydantic import BaseModel, Field

from config import settings as cfg


class CameraSettings(BaseModel):
    source: Union[int, str] = cfg.CAMERA_INDEX
    width: int = cfg.FRAME_WIDTH
    height: int = cfg.FRAME_HEIGHT
    grayscale: bool = cfg.GRAYSCALE
    fpsTarget: int = cfg.FPS_TARGET


class DetectionSettings(BaseModel):
    shuttleSource: Literal["local", "serverless", "off"] = cfg.SHUTTLE_SOURCE
    personConf: float = cfg.PERSON_CONFIDENCE
    ankleConf: float = cfg.ANKLE_CONFIDENCE
    shuttleConf: float = cfg.SHUTTLE_CONFIDENCE
    faceMatchThreshold: float = cfg.FACE_MATCH_THRESHOLD
    roboflowApiKey: str = ""


class DrillSettings(BaseModel):
    intervals: dict[str, float] = Field(
        default_factory=lambda: {
            "easy": cfg.DIFFICULTY["easy"]["interval"],
            "medium": cfg.DIFFICULTY["medium"]["interval"],
            "hard": cfg.DIFFICULTY["hard"]["interval"],
        }
    )
    defaultShots: int = 0
    returnConfirmFrames: int = cfg.RETURN_CONFIRM_FRAMES
    weakZoneThreshold: float = cfg.ZONE_WEAK_THRESHOLD


class CourtSettings(BaseModel):
    corners: list[list[float]] = Field(
        default_factory=lambda: [list(c) for c in cfg.COURT_CORNERS]
    )
    width: float = cfg.COURT_W
    length: float = cfg.COURT_L
    netDeadband: float = cfg.NET_DEADBAND


class Settings(BaseModel):
    camera: CameraSettings = Field(default_factory=CameraSettings)
    detection: DetectionSettings = Field(default_factory=DetectionSettings)
    drill: DrillSettings = Field(default_factory=DrillSettings)
    court: CourtSettings = Field(default_factory=CourtSettings)

    @classmethod
    def defaults(cls) -> "Settings":
        return cls()


class PlayerIn(BaseModel):
    name: str
    age: int = 0
    role: Literal["player", "staff", "admin"] = "player"
    jerseyNumber: int = 0
    imageUrl: str | None = None
    faceEnrolled: bool = False
    isActive: bool = True


class PlayerPatch(BaseModel):
    name: str | None = None
    age: int | None = None
    role: Literal["player", "staff", "admin"] | None = None
    jerseyNumber: int | None = None
    imageUrl: str | None = None
    faceEnrolled: bool | None = None
    isActive: bool | None = None


class SessionIn(BaseModel):
    title: str
    coachId: str | None = None
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    status: Literal["scheduled", "live", "completed", "cancelled"] = "scheduled"
    scheduledAt: str
    assignedPlayerIds: list[str] = Field(default_factory=list)
    notes: str = ""
    id: str | None = None


class SessionPatch(BaseModel):
    status: Literal["scheduled", "live", "completed", "cancelled"] | None = None
    title: str | None = None
    difficulty: Literal["easy", "medium", "hard"] | None = None
    notes: str | None = None
    assignedPlayerIds: list[str] | None = None

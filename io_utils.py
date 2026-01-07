import json
import os
from dataclasses import dataclass, asdict
from typing import Any, Dict, List

import pandas as pd


@dataclass
class OverlayConfig:
    show_possession_hud: bool = True
    show_clock: bool = True
    show_bboxes: bool = True
    show_track_ids: bool = True
    show_name_number: bool = True
    show_speed: bool = True
    show_distance: bool = True
    show_trails: bool = False
    show_leaderboard: bool = False
    show_pitch_mask: bool = False


@dataclass
class MatchConfig:
    team_a: str
    team_b: str
    half1_length_sec: int = 45 * 60
    half2_length_sec: int = 45 * 60
    halftime_break_min: int = 15
    pitch_length_m: float = 105.0
    pitch_width_m: float = 68.0
    possession_radius_m: float = 1.5
    possession_hysteresis_frames: int = 15
    detection_conf: float = 0.35
    device: str = "cuda"
    yolo_weights: str = "yolov8m.pt"


@dataclass
class CalibrationResult:
    pixel_points: Dict[str, List[float]]
    homography: List[List[float]]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def read_roster_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    expected = {"team", "name", "number", "position", "is_goalkeeper"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Roster CSV missing columns: {', '.join(sorted(missing))}")
    return df


def save_json(path: str, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_overlays(path: str, config: OverlayConfig) -> None:
    save_json(path, asdict(config))


def load_overlays(path: str) -> OverlayConfig:
    data = load_json(path)
    return OverlayConfig(**data)


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def save_dataframe_csv(path: str, df: pd.DataFrame) -> None:
    df.to_csv(path, index=False)

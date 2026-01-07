from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class TeamPrototypes:
    team_a: np.ndarray
    team_b: np.ndarray
    referee: np.ndarray


def compute_hsv_prototype(samples: List[np.ndarray]) -> np.ndarray:
    if not samples:
        return np.zeros(3)
    return np.mean(np.stack(samples, axis=0), axis=0)


def classify_hsv(hsv: np.ndarray, prototypes: TeamPrototypes) -> str:
    distances = {
        "team_a": np.linalg.norm(hsv - prototypes.team_a),
        "team_b": np.linalg.norm(hsv - prototypes.team_b),
        "referee": np.linalg.norm(hsv - prototypes.referee),
    }
    return min(distances, key=distances.get)


def smooth_prototypes(history: Dict[str, List[np.ndarray]], window: int = 20) -> TeamPrototypes:
    return TeamPrototypes(
        team_a=compute_hsv_prototype(history.get("team_a", [])[-window:]),
        team_b=compute_hsv_prototype(history.get("team_b", [])[-window:]),
        referee=compute_hsv_prototype(history.get("referee", [])[-window:]),
    )

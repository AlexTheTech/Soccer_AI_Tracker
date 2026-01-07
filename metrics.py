from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class PlayerMetrics:
    track_id: int
    team: str
    name: Optional[str] = None
    number: Optional[int] = None
    distance_m: float = 0.0
    max_speed_kmh: float = 0.0
    speeds: List[float] = field(default_factory=list)
    last_position: Optional[np.ndarray] = None

    @property
    def avg_speed_kmh(self) -> float:
        if not self.speeds:
            return 0.0
        return float(np.mean(self.speeds))


def update_metrics(metrics: Dict[int, PlayerMetrics],
                   track_id: int,
                   team: str,
                   position_m: np.ndarray,
                   dt: float,
                   name: Optional[str] = None,
                   number: Optional[int] = None) -> PlayerMetrics:
    if track_id not in metrics:
        metrics[track_id] = PlayerMetrics(track_id=track_id, team=team, name=name, number=number)
    player = metrics[track_id]
    if name:
        player.name = name
    if number is not None:
        player.number = number

    if player.last_position is not None:
        dist = float(np.linalg.norm(position_m - player.last_position))
        player.distance_m += dist
        speed_m_s = dist / dt if dt > 0 else 0.0
        speed_kmh = speed_m_s * 3.6
        player.speeds.append(speed_kmh)
        player.max_speed_kmh = max(player.max_speed_kmh, speed_kmh)
    player.last_position = position_m
    return player

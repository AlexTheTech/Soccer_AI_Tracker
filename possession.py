from collections import deque
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np


@dataclass
class PossessionState:
    current_owner: Optional[str] = None
    hysteresis_counter: int = 0


def assign_possession(ball_xy: Optional[np.ndarray],
                      player_positions: Dict[int, Tuple[np.ndarray, str]],
                      radius_m: float,
                      state: PossessionState,
                      hysteresis_frames: int) -> Optional[str]:
    if ball_xy is None or not player_positions:
        state.hysteresis_counter = max(0, state.hysteresis_counter - 1)
        return state.current_owner

    nearest_team = None
    nearest_dist = None
    for _, (pos, team) in player_positions.items():
        dist = float(np.linalg.norm(ball_xy - pos))
        if nearest_dist is None or dist < nearest_dist:
            nearest_dist = dist
            nearest_team = team

    if nearest_dist is None or nearest_dist > radius_m:
        state.hysteresis_counter = max(0, state.hysteresis_counter - 1)
        return state.current_owner

    if state.current_owner == nearest_team:
        state.hysteresis_counter = hysteresis_frames
        return state.current_owner

    if state.hysteresis_counter == 0:
        state.current_owner = nearest_team
        state.hysteresis_counter = hysteresis_frames
        return state.current_owner

    state.hysteresis_counter -= 1
    return state.current_owner

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from io_utils import OverlayConfig


def draw_hud(frame: np.ndarray,
             time_text: str,
             possession_text: str) -> None:
    cv2.rectangle(frame, (10, 10), (360, 80), (0, 0, 0), -1)
    cv2.putText(frame, time_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, possession_text, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 2)


def draw_track(frame: np.ndarray,
               bbox: np.ndarray,
               color: Tuple[int, int, int],
               label: str,
               show_bbox: bool) -> None:
    x1, y1, x2, y2 = bbox.astype(int)
    if show_bbox:
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, label, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def render_frame(frame: np.ndarray,
                 overlays: OverlayConfig,
                 tracks: List[Dict[str, object]],
                 possession_text: str,
                 time_text: str,
                 trail_history: Dict[int, List[Tuple[int, int]]],
                 leaderboard: Optional[List[str]] = None) -> np.ndarray:
    if overlays.show_clock or overlays.show_possession_hud:
        draw_hud(frame, time_text if overlays.show_clock else "",
                 possession_text if overlays.show_possession_hud else "")

    for track in tracks:
        color = track.get("color", (255, 255, 255))
        label = track.get("label", "")
        bbox = track.get("bbox")
        if bbox is None:
            continue
        draw_track(frame, np.array(bbox), color, label, overlays.show_bboxes)

    if overlays.show_trails:
        for track_id, points in trail_history.items():
            for idx in range(1, len(points)):
                cv2.line(frame, points[idx - 1], points[idx], (255, 255, 0), 2)

    if overlays.show_leaderboard and leaderboard:
        x0, y0 = 10, 90
        cv2.rectangle(frame, (x0, y0), (x0 + 240, y0 + 20 * len(leaderboard) + 10), (0, 0, 0), -1)
        for i, text in enumerate(leaderboard, start=1):
            cv2.putText(frame, text, (x0 + 10, y0 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return frame

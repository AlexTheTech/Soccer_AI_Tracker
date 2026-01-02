from dataclasses import dataclass
from types import SimpleNamespace
from typing import List, Optional, Tuple

import numpy as np

try:
    from ultralytics.trackers.byte_tracker import BYTETracker
    from ultralytics.trackers.utils import matching
except Exception:  # pragma: no cover
    BYTETracker = None
    matching = None


@dataclass
class Track:
    track_id: int
    bbox: np.ndarray
    cls: str
    conf: float


class Tracker:
    def __init__(self, fps: float, conf_thres: float = 0.35) -> None:
        self.conf_thres = conf_thres
        self.fps = fps
        self._tracker = None
        if BYTETracker is not None:
            try:
                args = SimpleNamespace(
                    track_thresh=conf_thres,
                    track_buffer=int(fps),
                    match_thresh=0.8,
                    aspect_ratio_thresh=1.6,
                    min_box_area=10,
                    mot20=False,
                )
                self._tracker = BYTETracker(args, frame_rate=fps)
            except TypeError:
                self._tracker = BYTETracker(
                    track_thresh=conf_thres,
                    track_buffer=int(fps),
                    match_thresh=0.8,
                    frame_rate=fps,
                )

    def update(self, detections: List[Track], frame_shape: Tuple[int, int]) -> List[Track]:
        if self._tracker is None:
            return self._simple_update(detections)
        if not detections:
            self._safe_update(np.empty((0, 5)), frame_shape)
            return []

        dets = np.array([np.append(d.bbox, d.conf) for d in detections], dtype=np.float32)
        online_targets = self._safe_update(dets, frame_shape)
        tracks: List[Track] = []
        for target in online_targets:
            if not hasattr(target, "tlbr"):
                continue
            bbox = np.array(target.tlbr, dtype=float)
            conf = float(getattr(target, "score", 1.0))
            cls_name = "player"
            if len(detections) > 0 and matching is not None:
                ious = matching.iou_batch(dets[:, :4], bbox[None, :])
                idx = int(np.argmax(ious))
                cls_name = detections[idx].cls
            tracks.append(Track(track_id=int(target.track_id), bbox=bbox, cls=cls_name, conf=conf))
        return tracks

    def _simple_update(self, detections: List[Track]) -> List[Track]:
        tracks: List[Track] = []
        for idx, det in enumerate(detections, start=1):
            tracks.append(Track(track_id=idx, bbox=det.bbox, cls=det.cls, conf=det.conf))
        return tracks

    def _safe_update(self, dets: np.ndarray, frame_shape: Tuple[int, int]):
        height, width = frame_shape
        try:
            return self._tracker.update(dets, (height, width), (height, width))
        except TypeError:
            return self._tracker.update(dets, np.empty((0,)), (height, width), (height, width))

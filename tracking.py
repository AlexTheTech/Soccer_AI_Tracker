from dataclasses import dataclass
from typing import List, Optional

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
            self._tracker = BYTETracker(
                track_thresh=conf_thres,
                track_buffer=int(fps),
                match_thresh=0.8,
                frame_rate=fps,
            )

    def update(self, detections: List[Track]) -> List[Track]:
        if self._tracker is None:
            return self._simple_update(detections)
        if not detections:
            self._tracker.update(np.empty((0, 5)), np.empty((0,)))
            return []

        dets = np.array([np.append(d.bbox, d.conf) for d in detections], dtype=np.float32)
        cls_ids = np.array([0 for _ in detections], dtype=np.float32)
        online_targets = self._tracker.update(dets, cls_ids)
        tracks: List[Track] = []
        for target in online_targets:
            if not hasattr(target, "tlbr"):
                continue
            bbox = np.array(target.tlbr, dtype=float)
            conf = float(getattr(target, "score", 1.0))
            idx = 0
            if len(detections) > 0:
                ious = matching.iou_batch(dets[:, :4], bbox[None, :])
                idx = int(np.argmax(ious))
            cls_name = detections[idx].cls if detections else "player"
            tracks.append(Track(track_id=int(target.track_id), bbox=bbox, cls=cls_name, conf=conf))
        return tracks

    def _simple_update(self, detections: List[Track]) -> List[Track]:
        tracks: List[Track] = []
        for idx, det in enumerate(detections, start=1):
            tracks.append(Track(track_id=idx, bbox=det.bbox, cls=det.cls, conf=det.conf))
        return tracks

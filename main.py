import json
import os
from collections import defaultdict
from typing import Dict, List, Optional

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from calibration import compute_homography, pixel_to_world
from detection import Detector, Detection
from gui import SoccerGUI
from io_utils import (CalibrationResult, MatchConfig, OverlayConfig, ensure_dir,
                      read_roster_csv, save_dataframe_csv, save_json, save_overlays,
                      write_jsonl)
from metrics import PlayerMetrics, update_metrics
from pitch_mask import compute_pitch_mask, is_point_in_mask
from possession import PossessionState, assign_possession
from render import render_frame
from team_role import TeamPrototypes, classify_hsv
from tracking import Track, Tracker


def _check_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Ensure CUDA drivers are installed.")


def _detection_to_track(detections: List[Detection]) -> List[Track]:
    tracks: List[Track] = []
    for det in detections:
        tracks.append(Track(track_id=-1, bbox=det.bbox, cls=det.cls, conf=det.conf))
    return tracks


def _load_prototypes(data: Dict[str, List[float]]) -> TeamPrototypes:
    return TeamPrototypes(
        team_a=np.array(data["team_a"], dtype=float),
        team_b=np.array(data["team_b"], dtype=float),
        referee=np.array(data["referee"], dtype=float),
    )


def _match_clock(frame_idx: int, fps: float, cfg: MatchConfig) -> str:
    seconds = frame_idx / fps
    half1 = cfg.half1_length_sec
    half2 = cfg.half2_length_sec
    halftime = cfg.halftime_break_min * 60
    if seconds <= half1:
        elapsed = seconds
        half_label = "H1"
    elif seconds <= half1 + halftime:
        elapsed = seconds - half1
        half_label = "HT"
    elif seconds <= half1 + halftime + half2:
        elapsed = seconds - half1 - halftime
        half_label = "H2"
    else:
        elapsed = seconds - half1 - halftime - half2
        half_label = "ET"
    mins = int(elapsed // 60)
    secs = int(elapsed % 60)
    return f"{half_label} {mins:02d}:{secs:02d}"


def process_video(video_path: str,
                  output_dir: str,
                  roster_path: str,
                  cfg: MatchConfig,
                  overlays: OverlayConfig,
                  calibration_points: Dict[str, tuple],
                  prototypes_data: Dict[str, List[float]],
                  manual_map: Dict[int, str]) -> None:
    _check_cuda()
    ensure_dir(output_dir)
    save_overlays(os.path.join(output_dir, "overlays.json"), overlays)

    roster = read_roster_csv(roster_path)

    detector = Detector(cfg.yolo_weights, device=cfg.device, conf=cfg.detection_conf)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Failed to open video")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    tracker = Tracker(fps=fps, conf_thres=cfg.detection_conf)
    homography = compute_homography(calibration_points, cfg.pitch_length_m, cfg.pitch_width_m)
    save_json(os.path.join(output_dir, "calibration.json"), {
        "pixel_points": calibration_points,
        "homography": homography.tolist(),
    })

    prototypes = _load_prototypes(prototypes_data)

    output_path = os.path.join(output_dir, "annotated_video.mp4")
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    metrics: Dict[int, PlayerMetrics] = {}
    possession_state = PossessionState()
    possession_counts = defaultdict(int)
    trail_history: Dict[int, List[tuple]] = defaultdict(list)
    frame_rows: List[Dict[str, object]] = []

    pitch_mask = None

    for frame_idx in tqdm(range(total_frames), desc="Processing"):
        ret, frame = cap.read()
        if not ret:
            break
        if pitch_mask is None:
            pitch_mask = compute_pitch_mask(frame)

        detections = detector.detect(frame)
        det_tracks = _detection_to_track(detections)
        tracks = tracker.update(det_tracks)

        ball_position_px: Optional[np.ndarray] = None
        player_world_positions: Dict[int, tuple] = {}
        render_tracks = []
        for trk in tracks:
            x1, y1, x2, y2 = trk.bbox
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            if pitch_mask is not None and not is_point_in_mask((cx, cy), pitch_mask):
                continue
            cls_lower = trk.cls.lower()
            if "ball" in cls_lower:
                ball_position_px = np.array([[cx, cy]], dtype=float)
                continue
            hsv = cv2.cvtColor(frame[int(max(0, cy-2)):int(min(height-1, cy+2)),
                                      int(max(0, cx-2)):int(min(width-1, cx+2))],
                                cv2.COLOR_BGR2HSV)
            if hsv.size == 0:
                continue
            hsv_mean = hsv.reshape(-1, 3).mean(axis=0)
            team = classify_hsv(hsv_mean, prototypes)
            if trk.track_id in manual_map:
                team = manual_map[trk.track_id]

            world_pos = pixel_to_world(np.array([[cx, cy]]), homography)[0]
            player_world_positions[trk.track_id] = (world_pos, team)

            roster_match = roster[roster["team"].str.lower() == (cfg.team_a.lower() if team == "team_a" else cfg.team_b.lower())]
            name = None
            number = None
            if not roster_match.empty:
                name = roster_match.iloc[0]["name"]
                number = roster_match.iloc[0]["number"]

            player_metrics = update_metrics(metrics, trk.track_id, team, world_pos, 1.0 / fps, name=name, number=number)
            label_parts = []
            if overlays.show_track_ids:
                label_parts.append(f"#{trk.track_id}")
            if overlays.show_name_number and player_metrics.name:
                label_parts.append(str(player_metrics.name))
            if overlays.show_speed:
                label_parts.append(f"{player_metrics.avg_speed_kmh:.1f} km/h")
            if overlays.show_distance:
                label_parts.append(f"{player_metrics.distance_m:.1f} m")
            label = " ".join(label_parts)
            color = (0, 255, 0) if team == "team_a" else (0, 0, 255)
            render_tracks.append({"bbox": trk.bbox, "label": label, "color": color})
            trail_history[trk.track_id].append((int(cx), int(cy)))
            if len(trail_history[trk.track_id]) > 30:
                trail_history[trk.track_id] = trail_history[trk.track_id][-30:]

        ball_world = None
        if ball_position_px is not None:
            ball_world = pixel_to_world(ball_position_px, homography)[0]

        owner = assign_possession(ball_world, player_world_positions,
                                  cfg.possession_radius_m, possession_state,
                                  cfg.possession_hysteresis_frames)
        if owner in ("team_a", "team_b"):
            possession_counts[owner] += 1

        total_poss = possession_counts["team_a"] + possession_counts["team_b"]
        bp_a = (possession_counts["team_a"] / total_poss * 100) if total_poss else 0.0
        bp_b = 100 - bp_a if total_poss else 0.0
        possession_text = f"{cfg.team_a}: {bp_a:.1f}% | {cfg.team_b}: {bp_b:.1f}%"
        time_text = _match_clock(frame_idx, fps, cfg)

        leaderboard = None
        if overlays.show_leaderboard:
            sorted_players = sorted(metrics.values(), key=lambda m: m.max_speed_kmh, reverse=True)[:5]
            leaderboard = [f"{m.name or m.track_id} {m.max_speed_kmh:.1f} km/h" for m in sorted_players]

        rendered = render_frame(frame.copy(), overlays, render_tracks, possession_text, time_text, trail_history, leaderboard)
        writer.write(rendered)

        frame_rows.append({
            "frame_idx": frame_idx,
            "timestamp": frame_idx / fps,
            "possession_owner": owner,
            "ball_world": ball_world.tolist() if ball_world is not None else None,
            "tracks": [
                {
                    "track_id": tid,
                    "team": team,
                    "world_x": float(pos[0]),
                    "world_y": float(pos[1]),
                }
                for tid, (pos, team) in player_world_positions.items()
            ],
        })

    cap.release()
    writer.release()

    players_df = pd.DataFrame([
        {
            "track_id": m.track_id,
            "team": m.team,
            "name": m.name,
            "number": m.number,
            "distance_m": m.distance_m,
            "max_speed_kmh": m.max_speed_kmh,
            "avg_speed_kmh": m.avg_speed_kmh,
        }
        for m in metrics.values()
    ])
    save_dataframe_csv(os.path.join(output_dir, "players_summary.csv"), players_df)

    team_distance = defaultdict(float)
    for m in metrics.values():
        team_distance[m.team] += m.distance_m

    total_poss = possession_counts["team_a"] + possession_counts["team_b"]
    bp_a = (possession_counts["team_a"] / total_poss * 100) if total_poss else 0.0
    bp_b = 100 - bp_a if total_poss else 0.0

    teams_df = pd.DataFrame([
        {
            "team": cfg.team_a,
            "BP": bp_a,
            "BPO": bp_b,
            "distance_m": team_distance.get("team_a", 0.0),
        },
        {
            "team": cfg.team_b,
            "BP": bp_b,
            "BPO": bp_a,
            "distance_m": team_distance.get("team_b", 0.0),
        },
    ])
    save_dataframe_csv(os.path.join(output_dir, "teams_summary.csv"), teams_df)

    write_jsonl(os.path.join(output_dir, "frame_tracks.jsonl"), frame_rows)


def main() -> None:
    gui = SoccerGUI()
    result = gui.run()
    process_video(
        result.video_path,
        result.output_dir,
        result.roster_path,
        result.match_config,
        result.overlays,
        result.calibration_points,
        result.prototypes,
        result.manual_track_map,
    )


if __name__ == "__main__":
    main()

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox

from calibration import compute_homography
from io_utils import MatchConfig, OverlayConfig, save_json


@dataclass
class GUIResult:
    video_path: str
    output_dir: str
    roster_path: str
    match_config: MatchConfig
    overlays: OverlayConfig
    calibration_points: Dict[str, Tuple[float, float]]
    calibration_mode: str
    calibration_side: str
    prototypes: Dict[str, List[float]]
    manual_track_map: Dict[int, str]


def _select_points(video_path: str, labels: List[str]) -> Dict[str, Tuple[float, float]]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Failed to open video for calibration")
    points: Dict[str, Tuple[float, float]] = {}
    frame_idx = 0

    def on_mouse(event, x, y, _flags, _param):
        nonlocal frame_idx
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < len(labels):
            label = labels[len(points)]
            points[label] = (float(x), float(y))
            print(f"Captured {label} at frame {frame_idx}: {points[label]}")

    window_name = "Calibration"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse)

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        display = frame.copy()
        if len(points) < len(labels):
            instruction = f"Step {len(points)+1}/{len(labels)}: Click {labels[len(points)]}"
        else:
            instruction = "Calibration complete"
        cv2.putText(display, instruction, (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(display, "Controls: n=next, p=prev, f=+10, b=-10, esc=cancel", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow(window_name, display)
        if len(points) >= len(labels):
            break
        key = cv2.waitKey(0)
        if key in (ord("n"), 83):
            continue
        if key in (ord("p"), 81):
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_idx - 2))
        if key == ord("f"):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx + 10)
        if key == ord("b"):
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_idx - 11))
        if key == 27:
            break

    cap.release()
    try:
        cv2.destroyWindow(window_name)
    except cv2.error:
        cv2.destroyAllWindows()
    if len(points) != len(labels):
        raise RuntimeError("Calibration cancelled")
    return points


def _sample_hsv(video_path: str, label: str, num_samples: int = 5) -> List[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Failed to open video for sampling")
    samples: List[np.ndarray] = []

    def on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN and len(samples) < num_samples:
            frame = current_frame.copy()
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            samples.append(hsv[y, x].astype(float))
            print(f"{label} sample {len(samples)}: {samples[-1]}")

    window_name = "Sampling"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse)
    current_frame = None
    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        current_frame = frame
        display = frame.copy()
        cv2.putText(display, f"Click {label} samples ({len(samples)}/{num_samples})", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(display, "Controls: n=next, p=prev, esc=cancel", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow(window_name, display)
        if len(samples) >= num_samples:
            break
        key = cv2.waitKey(0)
        if key == 27:
            break
        if key in (ord("n"), 83):
            continue
        if key in (ord("p"), 81):
            frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_idx - 2))
    cap.release()
    try:
        cv2.destroyWindow(window_name)
    except cv2.error:
        cv2.destroyAllWindows()
    if len(samples) < num_samples:
        raise RuntimeError("Sampling cancelled")
    return samples


class SoccerGUI:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Soccer AI Tracker")

        self.video_path = tk.StringVar()
        self.output_dir = tk.StringVar()
        self.roster_path = tk.StringVar()
        self.team_a = tk.StringVar(value="Team A")
        self.team_b = tk.StringVar(value="Team B")
        self.half1 = tk.StringVar(value="45:00")
        self.half2 = tk.StringVar(value="45:00")
        self.halftime = tk.StringVar(value="15")
        self.pitch_length = tk.StringVar(value="105")
        self.pitch_width = tk.StringVar(value="68")
        self.weights = tk.StringVar(value="yolov8m.pt")
        self.overlays = OverlayConfig()
        self.calibration_points: Optional[Dict[str, Tuple[float, float]]] = None
        self.prototypes: Optional[Dict[str, List[float]]] = None
        self.manual_map = tk.StringVar(value="")
        self.calibration_mode = tk.StringVar(value="full")
        self.partial_side = tk.StringVar(value="left")

        self._build()

    def _build(self) -> None:
        row = 0
        tk.Button(self.root, text="Select Video", command=self._select_video).grid(row=row, column=0, sticky="w")
        tk.Entry(self.root, textvariable=self.video_path, width=50).grid(row=row, column=1)
        row += 1

        tk.Button(self.root, text="Select Output Folder", command=self._select_output).grid(row=row, column=0, sticky="w")
        tk.Entry(self.root, textvariable=self.output_dir, width=50).grid(row=row, column=1)
        row += 1

        tk.Button(self.root, text="Select Roster CSV", command=self._select_roster).grid(row=row, column=0, sticky="w")
        tk.Entry(self.root, textvariable=self.roster_path, width=50).grid(row=row, column=1)
        row += 1

        tk.Label(self.root, text="Team A").grid(row=row, column=0, sticky="w")
        tk.Entry(self.root, textvariable=self.team_a).grid(row=row, column=1, sticky="w")
        row += 1

        tk.Label(self.root, text="Team B").grid(row=row, column=0, sticky="w")
        tk.Entry(self.root, textvariable=self.team_b).grid(row=row, column=1, sticky="w")
        row += 1

        tk.Label(self.root, text="Half 1 length (mm:ss)").grid(row=row, column=0, sticky="w")
        tk.Entry(self.root, textvariable=self.half1).grid(row=row, column=1, sticky="w")
        row += 1

        tk.Label(self.root, text="Half 2 length (mm:ss)").grid(row=row, column=0, sticky="w")
        tk.Entry(self.root, textvariable=self.half2).grid(row=row, column=1, sticky="w")
        row += 1

        tk.Label(self.root, text="Halftime break (min)").grid(row=row, column=0, sticky="w")
        tk.Entry(self.root, textvariable=self.halftime).grid(row=row, column=1, sticky="w")
        row += 1

        tk.Label(self.root, text="Pitch length/width (m)").grid(row=row, column=0, sticky="w")
        tk.Entry(self.root, textvariable=self.pitch_length, width=10).grid(row=row, column=1, sticky="w")
        tk.Entry(self.root, textvariable=self.pitch_width, width=10).grid(row=row, column=1)
        row += 1

        tk.Label(self.root, text="YOLO weights").grid(row=row, column=0, sticky="w")
        tk.Entry(self.root, textvariable=self.weights, width=50).grid(row=row, column=1, sticky="w")
        row += 1

        tk.Label(self.root, text="Calibration mode").grid(row=row, column=0, sticky="w")
        tk.Radiobutton(self.root, text="Full pitch (4 corners)", variable=self.calibration_mode,
                       value="full").grid(row=row, column=1, sticky="w")
        tk.Radiobutton(self.root, text="Partial pitch (side + midline)", variable=self.calibration_mode,
                       value="partial").grid(row=row, column=2, sticky="w")
        row += 1
        tk.Label(self.root, text="Partial side").grid(row=row, column=0, sticky="w")
        tk.Radiobutton(self.root, text="Left side", variable=self.partial_side, value="left").grid(row=row, column=1, sticky="w")
        tk.Radiobutton(self.root, text="Right side", variable=self.partial_side, value="right").grid(row=row, column=2, sticky="w")
        row += 1

        tk.Button(self.root, text="Calibration Wizard", command=self._calibrate).grid(row=row, column=0, sticky="w")
        tk.Button(self.root, text="Team/Ref Sampling", command=self._sample).grid(row=row, column=1, sticky="w")
        row += 1

        tk.Label(self.root, text="Manual track map (e.g. 12=team_a,15=referee)").grid(row=row, column=0, sticky="w")
        tk.Entry(self.root, textvariable=self.manual_map, width=50).grid(row=row, column=1, sticky="w")
        row += 1

        tk.Button(self.root, text="Overlay Options", command=self._overlay_dialog).grid(row=row, column=0, sticky="w")
        tk.Button(self.root, text="Start Processing", command=self.root.quit).grid(row=row, column=1, sticky="w")

    def _select_video(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.mov;*.mkv;*.avi")])
        if path:
            self.video_path.set(path)

    def _select_output(self) -> None:
        path = filedialog.askdirectory()
        if path:
            self.output_dir.set(path)

    def _select_roster(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if path:
            self.roster_path.set(path)

    def _calibrate(self) -> None:
        if not self.video_path.get():
            messagebox.showerror("Error", "Select a video first")
            return
        if self.calibration_mode.get() == "full":
            messagebox.showinfo(
                "Calibration Wizard",
                "Step through frames and click the four pitch corners in order:\n"
                "top_left, top_right, bottom_right, bottom_left.\n"
                "Controls: n=next, p=prev, f=+10, b=-10. Esc cancels.",
            )
            labels = ["top_left", "top_right", "bottom_right", "bottom_left"]
        else:
            side = self.partial_side.get()
            messagebox.showinfo(
                "Calibration Wizard",
                "Partial pitch mode:\n"
                f"Click {side} side corners and midfield points in order:\n"
                f"top_{side}, bottom_{side}, top_mid, bottom_mid.\n"
                "Controls: n=next, p=prev, f=+10, b=-10. Esc cancels.",
            )
            labels = [f"top_{side}", f"bottom_{side}", "top_mid", "bottom_mid"]
        self.calibration_points = _select_points(self.video_path.get(), labels)
        messagebox.showinfo("Calibration", "Calibration points captured")

    def _sample(self) -> None:
        if not self.video_path.get():
            messagebox.showerror("Error", "Select a video first")
            return
        messagebox.showinfo(
            "Team/Ref Sampling",
            "Click a few players for Team A, Team B, and Referee.\n"
            "Controls: n=next, p=prev. Esc cancels.",
        )
        samples = {}
        for label in ["team_a", "team_b", "referee"]:
            samples[label] = _sample_hsv(self.video_path.get(), label)
        self.prototypes = {
            key: np.mean(np.stack(vals, axis=0), axis=0).tolist() for key, vals in samples.items()
        }
        messagebox.showinfo("Sampling", "Color prototypes computed")

    def _overlay_dialog(self) -> None:
        dialog = tk.Toplevel(self.root)
        dialog.title("Overlays")
        vars_map = {
            "show_possession_hud": tk.BooleanVar(value=self.overlays.show_possession_hud),
            "show_clock": tk.BooleanVar(value=self.overlays.show_clock),
            "show_bboxes": tk.BooleanVar(value=self.overlays.show_bboxes),
            "show_track_ids": tk.BooleanVar(value=self.overlays.show_track_ids),
            "show_name_number": tk.BooleanVar(value=self.overlays.show_name_number),
            "show_speed": tk.BooleanVar(value=self.overlays.show_speed),
            "show_distance": tk.BooleanVar(value=self.overlays.show_distance),
            "show_trails": tk.BooleanVar(value=self.overlays.show_trails),
            "show_leaderboard": tk.BooleanVar(value=self.overlays.show_leaderboard),
            "show_pitch_mask": tk.BooleanVar(value=self.overlays.show_pitch_mask),
        }
        row = 0
        for label, var in vars_map.items():
            tk.Checkbutton(dialog, text=label, variable=var).grid(row=row, column=0, sticky="w")
            row += 1

        def save():
            for key, var in vars_map.items():
                setattr(self.overlays, key, bool(var.get()))
            dialog.destroy()

        tk.Button(dialog, text="Save", command=save).grid(row=row, column=0)

    def run(self) -> GUIResult:
        self.root.mainloop()
        if not self.video_path.get() or not self.output_dir.get() or not self.roster_path.get():
            raise RuntimeError("Missing required inputs")
        if not self.calibration_points:
            raise RuntimeError("Calibration is required")
        if not self.prototypes:
            raise RuntimeError("Team/ref sampling is required")

        half1 = _parse_time(self.half1.get())
        half2 = _parse_time(self.half2.get())
        halftime = int(self.halftime.get())
        pitch_length = float(self.pitch_length.get())
        pitch_width = float(self.pitch_width.get())

        match_config = MatchConfig(
            team_a=self.team_a.get(),
            team_b=self.team_b.get(),
            half1_length_sec=half1,
            half2_length_sec=half2,
            halftime_break_min=halftime,
            pitch_length_m=pitch_length,
            pitch_width_m=pitch_width,
            yolo_weights=self.weights.get(),
        )

        manual_map = _parse_manual_map(self.manual_map.get())
        return GUIResult(
            video_path=self.video_path.get(),
            output_dir=self.output_dir.get(),
            roster_path=self.roster_path.get(),
            match_config=match_config,
            overlays=self.overlays,
            calibration_points=self.calibration_points,
            calibration_mode=self.calibration_mode.get(),
            calibration_side=self.partial_side.get(),
            prototypes=self.prototypes,
            manual_track_map=manual_map,
        )


def _parse_time(value: str) -> int:
    parts = value.strip().split(":")
    if len(parts) != 2:
        raise ValueError("Time must be mm:ss")
    minutes = int(parts[0])
    seconds = int(parts[1])
    return minutes * 60 + seconds


def _parse_manual_map(value: str) -> Dict[int, str]:
    mapping: Dict[int, str] = {}
    if not value.strip():
        return mapping
    parts = value.split(",")
    for part in parts:
        if "=" not in part:
            continue
        track_str, team = part.split("=")
        mapping[int(track_str.strip())] = team.strip()
    return mapping

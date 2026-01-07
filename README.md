# Soccer AI Tracker

GPU-accelerated soccer match analytics for fixed elevated cameras (VEO-style). The application runs locally on your PC, takes a video file, and outputs an annotated video plus detailed CSV/JSON outputs.

## Features
- GUI-driven configuration (Tkinter)
- Calibration wizard (multi-frame corner selection)
- Team/ref color sampling for kit clashes
- YOLO detection + ByteTrack tracking
- Pitch masking to exclude spectators
- Ball possession and player metrics
- Annotated video with configurable overlays

## Setup

### 1) Create a Python environment
```bash
python -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies
```bash
pip install -r requirements.txt
```

### 3) Install PyTorch with CUDA
Follow the official instructions for your GPU:
- https://pytorch.org/get-started/locally/

Example (adjust CUDA version as needed):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## Run
```bash
python main.py
```

The GUI appears first. Select your inputs, complete calibration, sample kits, choose overlays, then start processing.

## Roster CSV format
The roster CSV must contain the following columns:
```
team,name,number,position,is_goalkeeper
```
Example:
```
Team A,Alex Smith,9,FW,false
Team B,Jordan Lee,1,GK,true
```

## Calibration Wizard (multi-frame)
- The full pitch does not need to be visible in one frame.
- Use the wizard to click the 4 pitch corners across any frames:
  - top_left, top_right, bottom_right, bottom_left
- You can advance frames with `n` and step back with `p`.
- The resulting homography maps pixel coordinates to meters based on pitch dimensions.

### Partial Pitch Mode
If the full pitch corners are never visible, switch to **Partial pitch** and choose a side:
- Left side: click `top_left`, `bottom_left`, `top_mid`, `bottom_mid`
- Right side: click `top_right`, `bottom_right`, `top_mid`, `bottom_mid`
This maps the side touchline plus midfield points to world coordinates.

## Team/Ref Sampling
- Click a few players for Team A, Team B, and a few referees across any frames.
- The system computes HSV prototypes and smooths classification.
- If needed, override a track ID in the GUI (e.g., `12=team_a,15=referee`).

## Outputs
All outputs are written to the chosen output folder:
- `annotated_video.mp4`
- `players_summary.csv`
- `teams_summary.csv`
- `frame_tracks.jsonl`
- `overlays.json`
- `calibration.json`

## Tuning
Key defaults (adjust in GUI or code):
- Detection confidence: 0.35
- Possession radius: 1.5 m
- Possession hysteresis: 15 frames
- Pitch dimensions: 105 x 68 m

## Visualization Options
- Enable **show_pitch_mask** in the overlay options to verify the pitch mask overlay is tracking the field correctly.

## Troubleshooting
- **CUDA not available**: Ensure NVIDIA drivers + CUDA-enabled PyTorch are installed.
- **Weights not found**: Provide a valid YOLO weights file path in the GUI (e.g., `yolov8m.pt`).
- **Ball lost**: The system handles ball loss gracefully; possession remains with the last owner until hysteresis expires.

## Project Structure
```
/gui.py
/main.py
/detection.py
/tracking.py
/pitch_mask.py
/calibration.py
/team_role.py
/possession.py
/metrics.py
/render.py
/io_utils.py
```

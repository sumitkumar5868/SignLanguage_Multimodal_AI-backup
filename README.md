# SignLanguage Multimodal AI

## Project Title
SignLanguage Multimodal AI

## Description
A software-based AI system that recognizes sign language and converts the recognized meaning into token/text, symbol, and sound outputs for improved accessibility.

## Current Status
Phase 5 — Custom Sign Language Dataset Collection

## Phase 2 — Webcam + OpenCV

Status: Completed

Pipeline:

```text
Laptop Camera
↓
OpenCV
↓
Python
↓
Live Video
```

The current phase establishes the live webcam feed. Sign-language recognition and AI processing will be implemented in later phases.

## Phase 3 — Hand Detection + Landmark Extraction

Status: Completed

Pipeline:

```text
Laptop Camera
↓
OpenCV
↓
Python
↓
MediaPipe Hand Landmarker
↓
Hand Detection
↓
21 Hand Landmarks
↓
Live Annotated Video
```

The system can now detect hands from the webcam, extract 21 hand landmark points, and display them on the live annotated video. Sign-language classification has NOT been implemented yet.

## Phase 4 — Landmark Coordinate Extraction

Status: Completed

Pipeline:

```text
Laptop Camera
↓
OpenCV
↓
MediaPipe Hand Landmarker
↓
21 Hand Landmarks
↓
X, Y, Z Coordinates
↓
63 Numerical Features
```

For one hand, 21 landmarks × 3 coordinates = 63 numerical features. This phase extracts the numerical landmark vector for verification only. Dataset collection and machine-learning training have NOT been implemented yet.

## Phase 5 — Custom Sign Language Dataset Collection

Status: **COMPLETED**

Purpose:
Collect labeled landmark-based sign language data using the webcam and MediaPipe. One-hand-only dataset collection in CSV format.

Pipeline:

```text
Laptop Camera
↓
OpenCV
↓
MediaPipe Hand Landmarker
↓
21 Hand Landmarks
↓
63 Numerical Features
↓
Selected Sign Label
↓
CSV Dataset
```

### Classes (Initial Set)

```text
HELLO
I_LOVE_YOU
I_HATE_YOU
I_EAT
THANK_YOU
```

### Target

500 samples per class × 5 classes = 2500 total samples.

### Dataset File

```text
dataset/sign_landmarks.csv
```

### Dataset Format

- **Columns**: 63 feature columns + 1 label column = 64 total
- **Features**: `x0,y0,z0,x1,y1,z1,...,x20,y20,z20`
- **Label**: `HELLO`, `I_LOVE_YOU`, `I_HATE_YOU`, `I_EAT`, `THANK_YOU`
- **Samples**: Numerical landmark coordinates, NOT images or videos

### Running Phase 5

```bash
py app.py
```

**Controls:**
- `1-5` — Select sign class
- `SPACE` — Capture one sample
- `Q` — Quit

**One-hand-only collection:**
- If exactly 1 hand is detected: SPACE saves the sample
- If 0 hands: Sample is NOT saved
- If 2+ hands: Sample is NOT saved

### Dataset Validation

```bash
py data/validate_dataset.py
```

This validates:
- CSV file exists
- Correct header (63 features + label)
- 64 total columns
- No missing values
- No invalid rows
- Five class distribution

### Machine Learning

Machine-learning training has NOT been implemented in Phase 5. Training and model prediction belong to Phase 6.

## Future Pipeline

```text
Webcam
↓
Hand/Sign Detection
↓
AI Sign Recognition
↓
Token + Symbol + Sound
↓
Accessibility Mode
```

This project is in the early foundation stage. Future phases will add computer vision, detection, recognition, and accessibility output modes step by step.

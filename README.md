# Watch Hand Leg Measurement System

Industrial computer-vision software for dimensional inspection of the **two tiny legs** on the backside of a watch hand / component.

The system uses **classical metrology techniques** (calibration, undistortion, ROI, thresholding, morphology, contours, rotated-rectangle / line fitting, optional sub-pixel refinement). It is **not** a generic object detector and does **not** use YOLO as the measurement engine.

Measurements are reported in **millimeters (mm)** after a validated pixel-to-mm calibration.

---

## Achievable accuracy (read this first)

This software converts **image measurements in pixels** to **physical units in millimeters**.

The achievable accuracy is **not a property of the software alone**. It depends on the optical and mechanical setup:

| Factor | Why it matters |
|---|---|
| Camera resolution | Smaller features need more pixels across the leg |
| Pixel size (sensor) | Sets the native sampling limit |
| Lens magnification / working distance | Determines mm-per-pixel at the object plane |
| Focus and depth of field | Soft edges inflate or shrink fitted dimensions |
| Lighting | Shadows and glare move apparent edges |
| Mechanical stability | Vibration and fixture play appear as measurement scatter |
| Calibration quality | Wrong scale factor scales every result |
| Part positioning / perspective | Tilt changes projected height and width |
| Optical distortion | Uncorrected barrel/pincushion error is geometric bias |

**Do not treat any micron-level accuracy claim as valid until the actual camera, lens, lighting, and fixture have been calibrated and verified against a traceable gauge.**

The UI always shows:

- pixel measurement
- converted mm measurement
- calibration factor (`mm/px`)
- calibration status: `CALIBRATED` / `NOT CALIBRATED`

Production inspection is **blocked** until a valid pixel-to-mm calibration exists.

---

## What is measured

| ID | Quantity | Method |
|---|---|---|
| 1 | Leg 1 height | Longitudinal side of `minAreaRect` (optional sub-pixel line fit) |
| 2 | Leg 1 width | Transverse (shorter) side of the fitted rectangle |
| 3 | Leg 2 height | Same as Leg 1 |
| 4 | Leg 2 width | Same as Leg 1 |
| 5 | Distance | Center-to-center and minimum edge-to-edge (Shapely) |
| 6 | Angle | Orientation of each rotated rectangle (optional) |
| 7 | Component size | Axis-aligned bounding box of the largest body contour, if present |

Width and height are taken **relative to the leg orientation**, not the image X/Y axes.

---

## Project layout

```
watch_leg_measurement/
├── app.py                 # Streamlit operator UI only
├── requirements.txt
├── README.md
├── config/config.yaml     # all tunable parameters
├── src/                   # computer vision / metrology engine
├── data/input             # images
├── data/calibration       # saved calibration YAML
├── data/output            # CSV / Excel reports
└── tests/
```

---

## Installation

Python **3.11+** is required.

```bash
cd watch_leg_measurement
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS
pip install -r requirements.txt
```

---

## Run the operator UI

```bash
streamlit run app.py
```

Typical workflow:

1. **Calibrate** — enter a known reference length (mm) and the corresponding pixel length, or run checkerboard camera calibration.
2. **Load image** — upload a file or pick one from `data/input`.
3. **Set ROI** — manual rectangle (sliders) or fixed ROI from `config.yaml`.
4. **Tune preprocessing** in the sidebar / `config.yaml` if the legs are not isolated.
5. **Run Inspection**.
6. Review overlay, tolerances, and **OK / NG**.
7. **Save Result** / **Export Excel**.

Buttons: `Run Inspection` · `Calibrate` · `Reset` · `Save Result` · `Export Excel`.

---

## Configuration

All thresholds live in `config/config.yaml`:

- camera backend
- pixel and camera calibration
- ROI
- preprocessing (blur, CLAHE, threshold method, morphology)
- segmentation
- leg-candidate filters (`min_area`, aspect ratio, …)
- pair-scoring weights
- sub-pixel measurement
- tolerances
- debug stage dumps

If the first images are not sufficient to lock a segmentation recipe, change `segmentation.method` (`adaptive` / `otsu` / `global` / `canny`) and the filter ranges rather than editing Python.

---

## Debug mode

Enable `debug.enabled` in the YAML or the UI checkbox. The UI then shows every stage:

1. Original  
2. ROI  
3. Grayscale  
4. Enhanced  
5. Threshold  
6. Morphology  
7. Edges  
8. Contours  
9. Candidate legs  
10. Final measurement  

Use this to tune threshold / morphology when the legs occupy only a few pixels.

---

## Industrial cameras (future)

Acquisition is abstracted behind `CameraInterface`:

```text
connect() / disconnect() / capture() / get_parameters()
```

The development backend is `ImageFileCamera`. Stub classes exist for Basler, Hikvision, IDS, and FLIR so a later SDK integration does not rewrite the measurement engine.

---

## Tests

```bash
pytest tests -q
```

Covered: pixel-to-mm conversion, camera-calibration I/O, Euclidean / edge-to-edge distance, rotated-box dimensions, tolerance evaluation.

---

## Reports

Each inspection appends a row:

`timestamp | image | calibration_factor | L1_width | L1_height | L2_width | L2_height | distance_c2c | distance_edge | result | status`

CSV and Excel paths are set under `reporting` in `config.yaml`.

---

## Design principles

- Deterministic: same image + same config → same result.
- No production measurement without calibration.
- Classical vision first; deep learning is not used for dimensioning.
- UI code is separated from the measurement engine (`src/`).
- Operator errors produce messages, not crashes.

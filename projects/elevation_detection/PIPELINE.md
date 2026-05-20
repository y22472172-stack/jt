# Bridge Elevation Detection Pipeline

This project pipeline turns a scanned bridge drawing PDF into structured pier
metrics and a local QA UI.

## Main Flow

1. PDF to enhanced page images
   - Script: `pdf_to_images.py`
   - Input: `GL5.102.AZ-507(1).pdf`
   - Output: `GL5.102.AZ-507(1)_images/page_0019.png`

2. Profile/content region detection and tiling
   - Script: `projects/elevation_detection/scripts/detect_page19_tiled.py`
   - Model: `projects/elevation_detection/models/elevation_detect_v2/weights/best.pt`
   - Output: `projects/elevation_detection/results/page_0019_vertical_segments_opt/page_0019_tiled_detections.json`

3. Pier region detection on profile crops
   - Script: `projects/elevation_detection/scripts/detect_214_on_profiles.py`
   - Model: `214/models/detect_214/weights/best.pt`
   - Output: `projects/elevation_detection/results/page_0019_detect_214_CD_expanded_profiles_v2/page_0019_detect_214_CD.json`

4. OCR and metric calculation
   - Script: `projects/elevation_detection/scripts/compute_pier_metrics_paddleocr.py`
   - OCR: PaddleOCR
   - Output: `projects/elevation_detection/results/page_0019_pier_metrics_final_run/page_0019_pier_metrics.json`
   - Metrics: pier number, top/middle/bottom elevations, pier height, embed depth, span groups, total length, lowest water level

5. QA store build
   - Script: `projects/elevation_detection/scripts/bridge_metrics_qa.py`
   - Output: `projects/elevation_detection/results/page_0019_pier_metrics_final_run/bridge_metrics_qa_store.json`

6. Visual QA UI
   - Script: `projects/elevation_detection/scripts/bridge_metrics_qa_server.py`
   - URL: `http://127.0.0.1:8765`

## Unified Runner

Run all non-interactive stages:

```powershell
c:\yw\conda\envs\mtorch\python.exe projects/elevation_detection/scripts/run_bridge_pipeline.py
```

Run only selected stages:

```powershell
c:\yw\conda\envs\mtorch\python.exe projects/elevation_detection/scripts/run_bridge_pipeline.py --skip-pdf --skip-ui
```

Start only the UI:

```powershell
c:\yw\conda\envs\mtorch\python.exe projects/elevation_detection/scripts/bridge_metrics_qa_server.py --host 127.0.0.1 --port 8765
```

## Retained Scripts

- `pdf_to_images.py`: PDF rendering and scan enhancement.
- `detect_page19_tiled.py`: profile/content region detection on the large page.
- `detect_214_on_profiles.py`: C/D pier-region detection on detected profile crops.
- `compute_pier_metrics_paddleocr.py`: OCR, global span/length parsing, pier metric calculation.
- `bridge_metrics_qa.py`: structured QA store and retrieval.
- `bridge_metrics_qa_server.py`: local visual QA interface.
- `run_bridge_pipeline.py`: unified pipeline runner.
- `xml_to_yolo.py`: dataset conversion utility, retained for retraining.

## Legacy or Removed Items

The old `scripts/` table/detail cropping utilities and early profile crop
experiments are not part of the current production pipeline. Historical metric
result folders under `projects/elevation_detection/results/page_0019_pier_metrics_*`
can be removed after `page_0019_pier_metrics_final_run` is kept.

Large training checkpoints such as `epoch*.pt` are not needed for inference.
Only `best.pt` is required by the current pipeline.

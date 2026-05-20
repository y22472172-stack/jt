"""
Run the bridge elevation detection pipeline with the current stable defaults.

The runner keeps each stage explicit so individual scripts remain usable.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
PYTHON = Path(sys.executable)

PDF = REPO_ROOT / "GL5.102.AZ-507(1).pdf"
PAGE_IMAGE = REPO_ROOT / "GL5.102.AZ-507(1)_images" / "page_0019.png"
PROFILE_OUT = REPO_ROOT / "projects" / "elevation_detection" / "results" / "page_0019_vertical_segments_opt"
PROFILE_JSON = PROFILE_OUT / "page_0019_tiled_detections.json"
PIER_OUT = REPO_ROOT / "projects" / "elevation_detection" / "results" / "page_0019_detect_214_CD_expanded_profiles_v2"
PIER_JSON = PIER_OUT / "page_0019_detect_214_CD.json"
METRIC_OUT = REPO_ROOT / "projects" / "elevation_detection" / "results" / "page_0019_pier_metrics_final_run"
METRIC_JSON = METRIC_OUT / "page_0019_pier_metrics.json"
QA_STORE = METRIC_OUT / "bridge_metrics_qa_store.json"


def run_command(args: list[str], cwd: Path = REPO_ROOT) -> None:
    print("\n> " + " ".join(args))
    subprocess.run(args, cwd=str(cwd), check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run bridge drawing OCR metric pipeline.")
    parser.add_argument("--skip-pdf", action="store_true", help="Skip PDF to image conversion.")
    parser.add_argument("--skip-profile", action="store_true", help="Skip profile detection.")
    parser.add_argument("--skip-pier", action="store_true", help="Skip pier region detection.")
    parser.add_argument("--skip-ocr", action="store_true", help="Skip OCR metric calculation.")
    parser.add_argument("--skip-qa", action="store_true", help="Skip QA store build.")
    parser.add_argument("--start-ui", action="store_true", help="Start local QA UI after building outputs.")
    parser.add_argument("--ocr-device", default="gpu:0")
    parser.add_argument("--no-gpu", action="store_true", help="Run PaddleOCR without GPU.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.skip_pdf and not PAGE_IMAGE.exists():
        run_command([str(PYTHON), "pdf_to_images.py"])
    elif args.skip_pdf or PAGE_IMAGE.exists():
        print(f"PDF stage skipped, using image: {PAGE_IMAGE}")

    if not args.skip_profile:
        run_command(
            [
                str(PYTHON),
                "projects/elevation_detection/scripts/detect_page19_tiled.py",
                "--image",
                str(PAGE_IMAGE),
                "--output-dir",
                str(PROFILE_OUT),
            ]
        )

    if not args.skip_pier:
        run_command(
            [
                str(PYTHON),
                "projects/elevation_detection/scripts/detect_214_on_profiles.py",
                "--image",
                str(PAGE_IMAGE),
                "--profile-json",
                str(PROFILE_JSON),
                "--output-dir",
                str(PIER_OUT),
            ]
        )

    if not args.skip_ocr:
        ocr_cmd = [
            str(PYTHON),
            "projects/elevation_detection/scripts/compute_pier_metrics_paddleocr.py",
            "--image",
            str(PAGE_IMAGE),
            "--detections",
            str(PIER_JSON),
            "--output-dir",
            str(METRIC_OUT),
        ]
        if not args.no_gpu:
            ocr_cmd.extend(["--use-gpu", "--ocr-device", args.ocr_device])
        run_command(ocr_cmd)

    if not args.skip_qa:
        run_command(
            [
                str(PYTHON),
                "projects/elevation_detection/scripts/bridge_metrics_qa.py",
                "build",
                "--input",
                str(METRIC_JSON),
                "--output",
                str(QA_STORE),
            ]
        )

    if args.start_ui:
        run_command(
            [
                str(PYTHON),
                "projects/elevation_detection/scripts/bridge_metrics_qa_server.py",
                "--host",
                "127.0.0.1",
                "--port",
                "8765",
                "--store",
                str(QA_STORE),
                "--result-json",
                str(METRIC_JSON),
            ]
        )

    print("\nPipeline outputs:")
    print(f"  profile detections: {PROFILE_JSON}")
    print(f"  pier detections:    {PIER_JSON}")
    print(f"  metric JSON:        {METRIC_JSON}")
    print(f"  QA store:           {QA_STORE}")


if __name__ == "__main__":
    main()

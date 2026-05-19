"""
Compute pier number, pier height, and embed depth from detected pier boxes.

The script uses the optimized detect_214 C/D results as pier anchors, then runs
PaddleOCR on a local crop around each pier. It extracts decimal elevations and
the circled pier number region, calculates:

    pier_height = top_elevation - middle_elevation
    embed_depth = middle_elevation - bottom_elevation
"""
from __future__ import annotations

import argparse
import csv
import importlib.metadata
import importlib.machinery
import json
import os
import re
import sys
import types
from pathlib import Path
from typing import Any

os.environ.setdefault("FLAGS_use_mkldnn", "0")
os.environ.setdefault("FLAGS_enable_mkldnn", "0")
os.environ.setdefault("FLAGS_use_onednn", "0")
os.environ.setdefault("FLAGS_enable_onednn", "0")
CONDA_DLL_DIR = Path(sys.prefix) / "Library" / "bin"
if os.name == "nt" and CONDA_DLL_DIR.exists():
    os.environ["PATH"] = f"{CONDA_DLL_DIR}{os.pathsep}{os.environ.get('PATH', '')}"
    os.add_dll_directory(str(CONDA_DLL_DIR))
NVIDIA_DLL_ROOT = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
if os.name == "nt" and NVIDIA_DLL_ROOT.exists():
    for nvidia_bin in NVIDIA_DLL_ROOT.glob("*\\bin"):
        os.environ["PATH"] = f"{nvidia_bin}{os.pathsep}{os.environ.get('PATH', '')}"
        os.add_dll_directory(str(nvidia_bin))

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_IMAGE = REPO_ROOT / "GL5.102.AZ-507(1)_images" / "page_0019.png"
DEFAULT_DETECTIONS = (
    REPO_ROOT
    / "projects"
    / "elevation_detection"
    / "results"
    / "page_0019_detect_214_CD_expanded_profiles_v2"
    / "page_0019_detect_214_CD.json"
)
DEFAULT_OUTPUT = (
    REPO_ROOT / "projects" / "elevation_detection" / "results" / "page_0019_pier_metrics"
)

STRICT_ELEVATION_RE = re.compile(r"[-一—]?\s*\d{1,3}[.,]\d{3}")
SHORT_DECIMAL_RE = re.compile(r"[-一—]?\s*\d{1,3}[.,]\d{1,2}")
SPACE_ELEVATION_RE = re.compile(r"[-一—]?\s*\d{1,3}\s+\d{3}")
COMPACT_ELEVATION_RE = re.compile(r"\b\d{4,5}\b")
INTEGER_RE = re.compile(r"\d{1,2}")
LW_NUMBER_RE = re.compile(r"L\s*W\s*[:：=]?\s*([-+]?\d{1,4}(?:[.,]\d{1,3})?)", re.IGNORECASE)
SPAN_RE = re.compile(r"(\d{1,2})\s*[xX×]\s*(\d{3,6})")
TOTAL_LENGTH_RE = re.compile(r"\b\d{5,6}\b")
CIRCLED_DIGITS = {
    "①": 1,
    "②": 2,
    "③": 3,
    "④": 4,
    "⑤": 5,
    "⑥": 6,
    "⑦": 7,
    "⑧": 8,
    "⑨": 9,
    "⑩": 10,
    "⑪": 11,
    "⑫": 12,
    "⑬": 13,
    "⑭": 14,
    "⑮": 15,
    "⑯": 16,
    "⑰": 17,
    "⑱": 18,
    "⑲": 19,
    "⑳": 20,
}


def normalize_number_text(text: str) -> str:
    normalized = text.strip()
    normalized = normalized.replace("一", "-").replace("—", "-").replace("－", "-")
    normalized = normalized.replace("。", ".").replace(",", ".")
    normalized = normalized.replace("O", "0").replace("o", "0")
    normalized = re.sub(r"\s+", "", normalized)
    return normalized


def add_unique_elevation(items: list[dict], value: float, quality: str, rule: str, token: str) -> None:
    for item in items:
        if abs(item["value"] - value) < 1e-6 and item["quality"] == quality and item["rule"] == rule:
            return
    items.append({"value": value, "quality": quality, "rule": rule, "token": token})


def parse_elevation_items(text: str) -> list[dict]:
    items: list[dict] = []
    consumed_spans: list[tuple[int, int]] = []

    for match in STRICT_ELEVATION_RE.finditer(text):
        token = normalize_number_text(match.group())
        if not re.fullmatch(r"-?\d{1,3}\.\d{3}", token):
            continue
        try:
            add_unique_elevation(items, float(token), "strict", "exact_3_decimal", token)
            consumed_spans.append(match.span())
        except ValueError:
            continue

    for match in SHORT_DECIMAL_RE.finditer(text):
        if any(start <= match.start() and match.end() <= end for start, end in consumed_spans):
            continue
        token = normalize_number_text(match.group())
        if not re.fullmatch(r"-?\d{1,3}\.\d{1,2}", token):
            continue
        sign = "-" if token.startswith("-") else ""
        body = token[1:] if sign else token
        integer, decimal = body.split(".")
        padded = f"{sign}{integer}.{decimal:<03}".replace(" ", "0")
        try:
            value = float(padded)
        except ValueError:
            continue
        add_unique_elevation(items, value, "corrected", "pad_decimal_to_3", token)
        if 0 <= value < 3:
            add_unique_elevation(items, value + 10, "corrected", "add_missing_leading_1", token)

    for match in SPACE_ELEVATION_RE.finditer(text):
        raw = match.group()
        normalized = raw.replace("一", "-").replace("—", "-").replace("－", "-")
        normalized = re.sub(r"\s+", " ", normalized.strip())
        sign = "-" if normalized.startswith("-") else ""
        unsigned = normalized[1:].strip() if sign else normalized
        parts = unsigned.split()
        if len(parts) != 2 or not parts[0].isdigit() or not parts[1].isdigit():
            continue
        token = f"{sign}{parts[0]}.{parts[1]}"
        try:
            value = float(token)
        except ValueError:
            continue
        add_unique_elevation(items, value, "corrected", "space_as_decimal_point", raw)
        if not sign and 8 <= value <= 35:
            add_unique_elevation(items, -value, "corrected", "space_as_decimal_point_missing_minus", raw)

    for match in COMPACT_ELEVATION_RE.finditer(text):
        if any(start <= match.start() and match.end() <= end for start, end in consumed_spans):
            continue
        token = match.group()
        if len(token) == 4:
            value_text = f"{token[0]}.{token[1:]}"
        else:
            value_text = f"{token[:2]}.{token[2:]}"
        try:
            value = float(value_text)
        except ValueError:
            continue
        add_unique_elevation(items, value, "corrected", "compact_integer_as_decimal", token)
    return items


def parse_elevations(text: str) -> list[float]:
    return [item["value"] for item in parse_elevation_items(text)]


def parse_pier_number(text: str) -> int | None:
    for char in text:
        if char in CIRCLED_DIGITS:
            return CIRCLED_DIGITS[char]

    match = INTEGER_RE.search(text)
    if match:
        value = int(match.group())
        if 1 <= value <= 99:
            return value
    return None


def parse_lw_number_text(text: str) -> tuple[float, str] | None:
    normalized = text.upper().replace(" ", "")
    normalized = normalized.replace("I", "L").replace("|", "L")
    normalized = normalized.replace("O", "0").replace("，", ".").replace(",", ".")
    match = LW_NUMBER_RE.search(normalized)
    if not match:
        return None

    token = match.group(1)
    if "." in token:
        try:
            return float(token), token
        except ValueError:
            return None

    if not token.lstrip("+-").isdigit():
        return None

    sign = "-" if token.startswith("-") else ""
    digits = token[1:] if sign else token
    if len(digits) == 2:
        value_text = f"{sign}{digits[0]}.{digits[1]}"
    elif len(digits) == 3:
        value_text = f"{sign}{digits[0]}.{digits[1:]}"
    elif len(digits) == 4:
        value_text = f"{sign}{digits[:2]}.{digits[2:]}"
    else:
        value_text = f"{sign}{digits}"
    try:
        return float(value_text), token
    except ValueError:
        return None


def clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def crop_with_bbox(img: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    return img[y1:y2, x1:x2].copy()


def make_local_bbox(
    pier_bbox: list[float],
    image_width: int,
    image_height: int,
    pad_x: int,
    pad_top: int,
    pad_bottom: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = [int(round(v)) for v in pier_bbox]
    return (
        clamp(x1 - pad_x, 0, image_width - 1),
        clamp(y1 - pad_top, 0, image_height - 1),
        clamp(x2 + pad_x, 1, image_width),
        clamp(y2 + pad_bottom, 1, image_height),
    )


def make_number_bbox(
    pier_bbox: list[float],
    image_width: int,
    image_height: int,
    pad_x: int,
    down: int,
) -> tuple[int, int, int, int]:
    x1, _, x2, y2 = [int(round(v)) for v in pier_bbox]
    center_x = (x1 + x2) // 2
    return (
        clamp(center_x - pad_x, 0, image_width - 1),
        clamp(y2 - 120, 0, image_height - 1),
        clamp(center_x + pad_x, 1, image_width),
        clamp(y2 + down, 1, image_height),
    )


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "0"


def major_version(version: str) -> int:
    match = re.match(r"(\d+)", version)
    return int(match.group(1)) if match else 0


class _TorchStubAny:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def __call__(self, *args: Any, **kwargs: Any) -> "_TorchStubAny":
        return _TorchStubAny()

    def __getattr__(self, name: str) -> Any:
        return _TorchStubAny

    def __iter__(self) -> Any:
        return iter(())


def install_torch_import_stub() -> None:
    if "torch" in sys.modules:
        return

    nn_stub = types.ModuleType("torch.nn")
    nn_stub.Module = object
    nn_stub.Linear = object
    nn_stub.__getattr__ = lambda name: object  # type: ignore[attr-defined]

    torch_stub = types.ModuleType("torch")
    torch_stub.__spec__ = importlib.machinery.ModuleSpec("torch", loader=None)
    torch_stub.__version__ = "0.0.0"
    torch_stub.Tensor = object
    torch_stub.device = str
    torch_stub.compile = lambda model, **kwargs: model
    torch_stub.no_grad = lambda: types.SimpleNamespace(
        __enter__=lambda self: None,
        __exit__=lambda self, *args: None,
    )
    torch_stub.cuda = types.SimpleNamespace(
        set_device=lambda *args, **kwargs: None,
        is_available=lambda: False,
    )
    torch_stub.multiprocessing = types.SimpleNamespace(
        get_start_method=lambda allow_none=True: "spawn",
        set_start_method=lambda *args, **kwargs: None,
    )
    torch_stub.distributed = types.SimpleNamespace(
        is_available=lambda: False,
        is_initialized=lambda: False,
        get_rank=lambda: 0,
        get_world_size=lambda: 1,
        init_process_group=lambda *args, **kwargs: None,
    )
    torch_stub.nn = nn_stub
    torch_stub.__getattr__ = lambda name: _TorchStubAny  # type: ignore[attr-defined]

    sys.modules["torch"] = torch_stub
    sys.modules["torch.nn"] = nn_stub
    sys.modules["torch.multiprocessing"] = torch_stub.multiprocessing
    sys.modules["torch.distributed"] = torch_stub.distributed


def create_paddle_ocr(args: argparse.Namespace) -> Any:
    version = package_version("paddleocr")
    major = major_version(version)
    print(f"PaddleOCR version: {version}")

    if args.use_gpu:
        try:
            import paddle

            if paddle.device.is_compiled_with_cuda():
                paddle.device.set_device(args.ocr_device)
                print(f"Paddle device: {paddle.device.get_device()}")
            else:
                print("Paddle is not compiled with CUDA; OCR will fall back to CPU.")
        except Exception as exc:
            print(f"Could not set Paddle GPU device: {exc}")

    if args.stub_torch_import and major >= 3:
        install_torch_import_stub()

    from paddleocr import PaddleOCR

    if major >= 3:
        kwargs = {
            "lang": args.lang,
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": False,
            "text_detection_model_name": args.text_detection_model_name,
            "text_recognition_model_name": args.text_recognition_model_name,
        }
        if args.use_gpu:
            kwargs["device"] = args.ocr_device
        return PaddleOCR(**kwargs)

    return PaddleOCR(
        lang=args.lang,
        use_angle_cls=True,
        show_log=False,
        use_gpu=args.use_gpu,
        enable_mkldnn=False,
    )


def run_paddle_ocr(ocr: Any, img: np.ndarray, scale: float = 2.0) -> list[dict]:
    if img.size == 0:
        return []

    if scale != 1.0:
        img_for_ocr = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    else:
        img_for_ocr = img

    if hasattr(ocr, "predict"):
        result = ocr.predict(img_for_ocr)
    else:
        result = ocr.ocr(img_for_ocr, cls=True)
    rows = []
    if not result:
        return rows

    if isinstance(result, list) and result and isinstance(result[0], dict):
        page = result[0]
        texts = page.get("rec_texts") or []
        scores = page.get("rec_scores") or []
        boxes = page.get("rec_boxes")
        if boxes is None:
            boxes = page.get("rec_polys")
        if boxes is None:
            boxes = []
        for text, score, box in zip(texts, scores, boxes):
            pts = np.array(box, dtype=np.float32)
            if pts.ndim == 1 and pts.size == 4:
                x1, y1, x2, y2 = pts.tolist()
            else:
                pts = pts.reshape(-1, 2)
                x1, y1 = pts[:, 0].min(), pts[:, 1].min()
                x2, y2 = pts[:, 0].max(), pts[:, 1].max()
            x1, y1, x2, y2 = [value / scale for value in [x1, y1, x2, y2]]
            rows.append(
                {
                    "text": str(text),
                    "confidence": float(score),
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "center": [float((x1 + x2) / 2), float((y1 + y2) / 2)],
                }
            )
        return rows

    # PaddleOCR 2.x returns [ [box, (text, score)], ... ] inside a page list.
    lines = result[0] if len(result) == 1 and isinstance(result[0], list) else result
    for line in lines:
        if not line or len(line) < 2:
            continue
        box, payload = line[0], line[1]
        if not isinstance(payload, (tuple, list)) or len(payload) < 2:
            continue
        text = str(payload[0])
        score = float(payload[1])
        pts = np.array(box, dtype=np.float32) / scale
        x1, y1 = pts[:, 0].min(), pts[:, 1].min()
        x2, y2 = pts[:, 0].max(), pts[:, 1].max()
        rows.append(
            {
                "text": text,
                "confidence": score,
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "center": [float((x1 + x2) / 2), float((y1 + y2) / 2)],
            }
        )
    return rows


def extract_elevation_candidates(ocr_rows: list[dict], origin: tuple[int, int]) -> list[dict]:
    ox, oy = origin
    candidates = []
    for row in ocr_rows:
        parsed_items = parse_elevation_items(row["text"])
        for parsed in parsed_items:
            value = parsed["value"]
            x1, y1, x2, y2 = row["bbox"]
            cx, cy = row["center"]
            candidates.append(
                {
                    "value": value,
                    "text": row["text"],
                    "candidate_quality": parsed["quality"],
                    "source_rule": parsed["rule"],
                    "token": parsed["token"],
                    "corrected": parsed["quality"] == "corrected",
                    "confidence": row["confidence"] if parsed["quality"] == "strict" else row["confidence"] * 0.82,
                    "local_bbox": row["bbox"],
                    "bbox": [x1 + ox, y1 + oy, x2 + ox, y2 + oy],
                    "center": [cx + ox, cy + oy],
                }
            )
    return candidates


def extract_lowest_water_level(global_ocr_rows: list[dict], origin: tuple[int, int]) -> dict | None:
    ox, oy = origin
    candidates = []
    for row in global_ocr_rows:
        parsed = parse_lw_number_text(row["text"])
        if not parsed:
            continue
        value, token = parsed
        x1, y1, x2, y2 = row["bbox"]
        cx, cy = row["center"]
        candidates.append(
            {
                "value": value,
                "token": token,
                "text": row["text"],
                "confidence": row["confidence"],
                "bbox": [x1 + ox, y1 + oy, x2 + ox, y2 + oy],
                "center": [cx + ox, cy + oy],
                "source": row.get("source_image", "global_ocr"),
            }
        )

    if not candidates:
        return None
    return max(candidates, key=lambda item: item["confidence"])


def row_to_global_item(row: dict, origin: tuple[int, int]) -> dict:
    ox, oy = origin
    x1, y1, x2, y2 = row["bbox"]
    cx, cy = row["center"]
    return {
        "text": row["text"],
        "confidence": row["confidence"],
        "bbox": [x1 + ox, y1 + oy, x2 + ox, y2 + oy],
        "center": [cx + ox, cy + oy],
        "source": row.get("source_image", "global_ocr"),
    }


def span_count_candidates(raw_count: int, confidence: float) -> list[dict]:
    candidates = [
        {
            "span_count": raw_count,
            "confidence": confidence,
            "penalty": 0.0,
            "reason": "ocr",
        }
    ]
    # OCR frequently confuses these digits in small engineering labels. These
    # are alternatives for global consistency scoring, not unconditional fixes.
    confusions = {
        9: [(3, 0.45), (4, 0.8)],
        8: [(3, 0.65)],
        6: [(3, 0.75), (5, 0.75)],
        5: [(3, 0.85)],
    }
    for count, penalty in confusions.get(raw_count, []):
        candidates.append(
            {
                "span_count": count,
                "confidence": max(0.0, confidence - penalty),
                "penalty": penalty + max(0.0, confidence - 0.92) * 0.35,
                "reason": f"ocr_confusion_{raw_count}_to_{count}",
            }
        )
    return candidates


def correct_span_counts_by_pier_count(groups: list[dict], pier_count: int | None) -> None:
    if not groups:
        return

    candidate_sets = [
        span_count_candidates(item["raw_span_count"], item["confidence"])
        for item in groups
    ]
    combinations: list[tuple[float, list[dict], int]] = [(0.0, [], 0)]
    for candidate_set in candidate_sets:
        next_combinations = []
        for score, chosen, total in combinations:
            for candidate in candidate_set:
                next_combinations.append(
                    (
                        score + candidate["penalty"],
                        chosen + [candidate],
                        total + candidate["span_count"],
                    )
                )
        combinations = next_combinations

    if pier_count:
        exact = [combo for combo in combinations if combo[2] == pier_count]
        if exact:
            _, chosen, _ = min(exact, key=lambda combo: combo[0])
        else:
            chosen = min(
                combinations,
                key=lambda combo: (abs(combo[2] - pier_count), combo[0]),
            )[1]
    else:
        chosen = min(combinations, key=lambda combo: combo[0])[1]

    for item, candidate in zip(groups, chosen):
        item["span_count"] = candidate["span_count"]
        item["span_count_confidence"] = candidate["confidence"]
        item["span_count_reason"] = candidate["reason"]
        if item["span_count"] != item["raw_span_count"]:
            item["span_count_corrected"] = True
            item["correction_reason"] = (
                f"{candidate['reason']};"
                f"span_count_sum_matches_pier_count_{pier_count}"
                if pier_count and sum(c["span_count"] for c in chosen) == pier_count
                else candidate["reason"]
            )


def extract_span_groups(
    global_ocr_rows: list[dict],
    origin: tuple[int, int],
    start_number: int,
    pier_count: int | None = None,
) -> list[dict]:
    raw_groups = []
    for row in global_ocr_rows:
        match = SPAN_RE.search(row["text"])
        if not match:
            continue
        span_count = int(match.group(1))
        span_length = int(match.group(2))
        item = row_to_global_item(row, origin)
        item.update(
            {
                "raw_span_count": span_count,
                "span_count": span_count,
                "span_length": span_length,
            }
        )
        raw_groups.append(item)

    raw_groups.sort(key=lambda item: (item["center"][0], -item["confidence"]))
    groups = []
    for item in raw_groups:
        if groups and abs(item["center"][0] - groups[-1]["center"][0]) < 140:
            if item["confidence"] > groups[-1]["confidence"]:
                groups[-1] = item
            continue
        groups.append(item)

    correct_span_counts_by_pier_count(groups, pier_count)

    next_pier = start_number
    for index, item in enumerate(groups, start=1):
        span_count = item["span_count"]
        item["span_group_index"] = index
        item["pier_indices"] = list(range(next_pier, next_pier + span_count))
        item["description"] = f"{span_count} spans @ {span_length}"
        next_pier += span_count
    return groups


def extract_total_length(global_ocr_rows: list[dict], origin: tuple[int, int], min_value: int = 30000) -> dict | None:
    candidates = []
    for row in global_ocr_rows:
        text = normalize_number_text(row["text"])
        match = TOTAL_LENGTH_RE.search(text)
        if not match:
            continue
        value = int(match.group())
        if value < min_value:
            continue
        item = row_to_global_item(row, origin)
        item.update({"value": value, "token": match.group(), "unit": "drawing_units"})
        candidates.append(item)

    if not candidates:
        return None

    return max(candidates, key=lambda item: (item["confidence"], -item["center"][1]))


def add_source_to_rows(rows: list[dict], source: str) -> list[dict]:
    tagged = []
    for row in rows:
        item = dict(row)
        item["source_image"] = source
        tagged.append(item)
    return tagged


def make_ocr_variants(img: np.ndarray) -> list[tuple[str, np.ndarray, float]]:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    variants: list[tuple[str, np.ndarray, float]] = [("original", img, 1.0)]

    scale = 3.0
    up = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    blurred = cv2.GaussianBlur(up, (0, 0), 1.1)
    sharp = cv2.addWeighted(up, 1.7, blurred, -0.7, 0)
    variants.append(("sharp_x3", sharp, scale))

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    binary = cv2.adaptiveThreshold(
        clahe,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        9,
    )
    binary_bgr = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    binary_up = cv2.resize(binary_bgr, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
    variants.append(("binary_contrast_x2_5", binary_up, 2.5))
    return variants


def run_paddle_ocr_preprocessed(ocr: Any, img: np.ndarray, scale_back: float) -> list[dict]:
    rows = run_paddle_ocr(ocr, img, scale=1.0)
    if scale_back == 1.0:
        return rows
    for row in rows:
        row["bbox"] = [float(v / scale_back) for v in row["bbox"]]
        row["center"] = [float(v / scale_back) for v in row["center"]]
    return rows


def merge_ocr_rows(rows: list[dict], iou_threshold: float = 0.55) -> list[dict]:
    def iou(a: list[float], b: list[float]) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        denom = area_a + area_b - inter
        return inter / denom if denom > 0 else 0.0

    merged: list[dict] = []
    for row in sorted(rows, key=lambda item: item.get("confidence", 0), reverse=True):
        duplicate = False
        for kept in merged:
            same_text = normalize_number_text(row["text"]) == normalize_number_text(kept["text"])
            close_center = (
                abs(row["center"][0] - kept["center"][0]) < 30
                and abs(row["center"][1] - kept["center"][1]) < 18
            )
            if same_text and (iou(row["bbox"], kept["bbox"]) >= iou_threshold or close_center):
                duplicate = True
                kept.setdefault("merged_sources", []).append(row.get("source_image", "unknown"))
                break
        if not duplicate:
            item = dict(row)
            item["merged_sources"] = [row.get("source_image", "unknown")]
            merged.append(item)
    return sorted(merged, key=lambda item: (item["center"][1], item["center"][0]))


def make_global_ocr_bbox(
    pier_boxes: list[dict],
    image_width: int,
    image_height: int,
    pad_x: int,
    pad_top: int,
    pad_bottom: int,
) -> tuple[int, int, int, int]:
    xs1 = [det["bbox"][0] for det in pier_boxes]
    ys1 = [det["bbox"][1] for det in pier_boxes]
    xs2 = [det["bbox"][2] for det in pier_boxes]
    ys2 = [det["bbox"][3] for det in pier_boxes]
    return (
        clamp(int(round(min(xs1) - pad_x)), 0, image_width - 1),
        clamp(int(round(min(ys1) - pad_top)), 0, image_height - 1),
        clamp(int(round(max(xs2) + pad_x)), 1, image_width),
        clamp(int(round(max(ys2) + pad_bottom)), 1, image_height),
    )


def run_global_multivariant_ocr(
    ocr: Any,
    img: np.ndarray,
    bbox: tuple[int, int, int, int],
    debug_dir: Path,
    tile_width: int,
    tile_overlap: int,
) -> tuple[list[dict], list[dict]]:
    crop = crop_with_bbox(img, bbox)
    cv2.imwrite(str(debug_dir / "global_ocr_crop.png"), crop)
    all_rows: list[dict] = []
    crop_h, crop_w = crop.shape[:2]
    step = max(1, tile_width - tile_overlap)
    tile_starts = list(range(0, max(1, crop_w), step))
    if tile_starts and tile_starts[-1] + tile_width >= crop_w:
        tile_starts = [start for start in tile_starts if start < crop_w]

    for tile_index, start_x in enumerate(tile_starts, start=1):
        end_x = min(crop_w, start_x + tile_width)
        if end_x - start_x < min(400, tile_width // 3) and start_x > 0:
            continue
        tile = crop[:, start_x:end_x]
        for name, variant, scale_back in make_ocr_variants(tile):
            if tile_index <= 3:
                cv2.imwrite(str(debug_dir / f"global_tile_{tile_index:02d}_{name}.png"), variant)
            rows = run_paddle_ocr_preprocessed(ocr, variant, scale_back=scale_back)
            for row in rows:
                row["bbox"] = [
                    row["bbox"][0] + start_x,
                    row["bbox"][1],
                    row["bbox"][2] + start_x,
                    row["bbox"][3],
                ]
                row["center"] = [row["center"][0] + start_x, row["center"][1]]
            all_rows.extend(add_source_to_rows(rows, f"{name}:tile_{tile_index:02d}"))

    merged_rows = merge_ocr_rows(all_rows)
    candidates = extract_elevation_candidates(merged_rows, (bbox[0], bbox[1]))
    for candidate in candidates:
        candidate["global_source"] = "global_multivariant"
    return merged_rows, candidates


def cluster_elevation_bands(candidates: list[dict], y_tolerance: float = 80.0) -> list[dict]:
    useful = [c for c in candidates if -35 <= c["value"] <= 25]
    clusters: list[list[dict]] = []
    for candidate in sorted(useful, key=lambda item: item["center"][1]):
        placed = False
        for cluster in clusters:
            center_y = sum(c["center"][1] for c in cluster) / len(cluster)
            if abs(candidate["center"][1] - center_y) <= y_tolerance:
                cluster.append(candidate)
                placed = True
                break
        if not placed:
            clusters.append([candidate])

    bands = []
    for index, cluster in enumerate(clusters, start=1):
        values = [c["value"] for c in cluster]
        center_y = sum(c["center"][1] for c in cluster) / len(cluster)
        positives = sum(1 for value in values if value >= 0)
        negatives = len(values) - positives
        if negatives > positives:
            role = "bottom"
        elif np.median(values) >= 13:
            role = "top"
        else:
            role = "middle"
        bands.append(
            {
                "band_id": index,
                "center_y": float(center_y),
                "role": role,
                "count": len(cluster),
                "min_value": float(min(values)),
                "max_value": float(max(values)),
            }
        )
        for candidate in cluster:
            candidate["band_id"] = index
            candidate["band_role"] = role
    return bands


def pier_search_window(pier_boxes: list[dict], idx: int, margin: float) -> tuple[float, float]:
    centers = [((det["bbox"][0] + det["bbox"][2]) / 2) for det in pier_boxes]
    center = centers[idx]
    left = (centers[idx - 1] + center) / 2 if idx > 0 else pier_boxes[idx]["bbox"][0] - margin
    right = (center + centers[idx + 1]) / 2 if idx < len(centers) - 1 else pier_boxes[idx]["bbox"][2] + margin
    return left - margin, right + margin


def candidate_in_window(candidate: dict, left: float, right: float) -> bool:
    x1, _, x2, _ = candidate["bbox"]
    cx = candidate["center"][0]
    return left <= cx <= right or (x1 <= right and x2 >= left)


def select_elevations_with_constraints(
    global_candidates: list[dict],
    local_candidates: list[dict],
    pier_boxes: list[dict],
    pier_index: int,
    args: argparse.Namespace,
) -> dict:
    det = pier_boxes[pier_index]
    x1, y1, x2, y2 = det["bbox"]
    center_x = (x1 + x2) / 2
    left, right = pier_search_window(pier_boxes, pier_index, args.global_match_margin)
    pool = [c for c in global_candidates if candidate_in_window(c, left, right)]
    if args.use_local_fallback:
        pool.extend(local_candidates)

    top_pool = [c for c in pool if args.top_min <= c["value"] <= args.top_max]
    middle_pool = [c for c in pool if args.middle_min <= c["value"] <= args.middle_max]
    bottom_pool = [c for c in pool if args.bottom_min <= c["value"] < 0]

    def score(role: str, candidate: dict) -> float:
        cx, cy = candidate["center"]
        role_target_y = {
            "top": y1 + (y2 - y1) * 0.06,
            "middle": y1 + (y2 - y1) * 0.36,
            "bottom": y1 + (y2 - y1) * 0.92,
        }[role]
        band_bonus = 0
        if candidate.get("band_role") == role:
            band_bonus -= 80
        elif candidate.get("band_role"):
            band_bonus += 80
        source_bonus = -20 if candidate.get("global_source") else 0
        correction_penalty = args.corrected_candidate_penalty if candidate.get("candidate_quality") == "corrected" else 0
        y_weight = args.bottom_y_weight if role == "bottom" else args.band_y_weight
        value_bias = 0.0
        if role == "bottom":
            value_bias = candidate["value"] * args.bottom_value_weight
        return (
            abs(cx - center_x) * args.x_weight
            + abs(cy - role_target_y) * y_weight
            - candidate["confidence"] * 35
            + band_bonus
            + source_bonus
            + correction_penalty
            + value_bias
        )

    def combo_geometry_score(top: dict, middle: dict, bottom: dict) -> float | None:
        top_x, top_y = top["center"]
        middle_x, middle_y = middle["center"]
        bottom_x, bottom_y = bottom["center"]

        if not top_y + args.vertical_order_tolerance < middle_y:
            return None
        if not middle_y + args.vertical_order_tolerance < bottom_y:
            return None

        xs = [top_x, middle_x, bottom_x]
        x_spread = max(xs) - min(xs)
        if x_spread > args.max_elevation_x_spread:
            return None

        midline_x = sum(xs) / 3
        x_alignment = sum(abs(x - midline_x) for x in xs)

        top_mid_gap = middle_y - top_y
        mid_bottom_gap = bottom_y - middle_y
        ratio_penalty = abs(top_mid_gap - mid_bottom_gap) * args.vertical_spacing_weight

        return x_alignment * args.alignment_x_weight + ratio_penalty

    best_combo = None
    best_score = float("inf")
    for top in top_pool:
        for middle in middle_pool:
            if not top["value"] > middle["value"]:
                continue
            height = top["value"] - middle["value"]
            if not args.min_pier_height <= height <= args.max_pier_height:
                continue
            for bottom in bottom_pool:
                if not middle["value"] > bottom["value"]:
                    continue
                embed = middle["value"] - bottom["value"]
                if not args.min_embed_depth <= embed <= args.max_embed_depth:
                    continue
                combo_score = score("top", top) + score("middle", middle) + score("bottom", bottom)
                geometry_score = combo_geometry_score(top, middle, bottom)
                if args.require_geometry and geometry_score is None:
                    continue
                if geometry_score is not None:
                    combo_score += geometry_score
                if best_combo is None or combo_score < best_score:
                    best_combo = (top, middle, bottom)
                    best_score = combo_score

    if best_combo:
        top, middle, bottom = best_combo
        return {
            "top": top,
            "middle": middle,
            "bottom": bottom,
            "match_status": "valid_global_constraint",
            "search_window": [left, right],
        }

    # Return partial constrained matches for review, but never violate bottom < 0
    top = min(top_pool, key=lambda c: score("top", c), default=None)
    middle_candidates = [
        c
        for c in middle_pool
        if top is None or top["value"] > c["value"]
    ]
    middle = min(middle_candidates, key=lambda c: score("middle", c), default=None)
    bottom_candidates = [
        c
        for c in bottom_pool
        if middle is None or middle["value"] > c["value"]
    ]
    bottom = min(bottom_candidates, key=lambda c: score("bottom", c), default=None)
    return {
        "top": top,
        "middle": middle,
        "bottom": bottom,
        "match_status": "partial_constraint",
        "search_window": [left, right],
    }


def fmt_value(value: float | None) -> str:
    return f"{value:.3f}" if value is not None else ""


def clone_inferred_candidate(candidate: dict, source: str, center: tuple[float, float] | None = None) -> dict:
    inferred = dict(candidate)
    inferred["source_rule"] = source
    inferred["confidence"] = min(float(inferred.get("confidence", 0.0)), 0.5)
    if center is not None:
        old_x, old_y = inferred["center"]
        new_x, new_y = center
        dx = new_x - old_x
        dy = new_y - old_y
        inferred["center"] = [float(new_x), float(new_y)]
        inferred["bbox"] = [
            float(inferred["bbox"][0] + dx),
            float(inferred["bbox"][1] + dy),
            float(inferred["bbox"][2] + dx),
            float(inferred["bbox"][3] + dy),
        ]
        inferred["position_inferred"] = True
    return inferred


def inferred_candidate_center(item: dict, key: str) -> tuple[float, float]:
    selected = item["selected_elevations"]
    available_x = [
        selected[other]["center"][0]
        for other in ["top", "middle", "bottom"]
        if selected.get(other)
    ]
    if available_x:
        target_x = float(np.median(available_x))
    else:
        x1, _, x2, _ = item["pier_bbox"]
        target_x = float((x1 + x2) / 2)

    _, y1, _, y2 = item["pier_bbox"]
    role_ratio = {"top": 0.06, "middle": 0.36, "bottom": 0.92}[key]
    target_y = float(y1 + (y2 - y1) * role_ratio)
    return target_x, target_y


def nearest_valid_candidate(results: list[dict], index: int, key: str) -> dict | None:
    best = None
    best_distance = float("inf")
    for other_index, item in enumerate(results):
        if other_index == index:
            continue
        if "valid_global_constraint" not in item.get("status", ""):
            continue
        candidate = item["selected_elevations"].get(key)
        if not candidate:
            continue
        distance = abs(other_index - index)
        if distance < best_distance:
            best = candidate
            best_distance = distance
    return best


def valid_metric_triplet(top: dict | None, middle: dict | None, bottom: dict | None, args: argparse.Namespace) -> bool:
    if not (top and middle and bottom):
        return False
    if not top["value"] > middle["value"] > bottom["value"]:
        return False
    if not bottom["value"] < 0:
        return False
    return (
        args.min_pier_height <= top["value"] - middle["value"] <= args.max_pier_height
        and args.min_embed_depth <= middle["value"] - bottom["value"] <= args.max_embed_depth
    )


def postprocess_results(results: list[dict], args: argparse.Namespace) -> None:
    for offset, item in enumerate(results):
        expected_number = args.start_number + offset
        original = item["pier_number"]
        item["pier_number"] = {
            "number": expected_number,
            "text": original.get("text"),
            "confidence": original.get("confidence", 0.0),
            "bbox": original.get("bbox"),
            "source": "sequential_left_to_right",
            "ocr_number": original.get("number"),
        }

    for index, item in enumerate(results):
        selected = item["selected_elevations"]
        top = selected.get("top")
        middle = selected.get("middle")
        bottom = selected.get("bottom")
        inferred_keys = []

        for key in ["top", "middle", "bottom"]:
            if selected.get(key):
                continue
            neighbor = nearest_valid_candidate(results, index, key)
            if neighbor:
                selected[key] = clone_inferred_candidate(
                    neighbor,
                    f"nearest_valid_{key}",
                    inferred_candidate_center(item, key),
                )
                inferred_keys.append(key)

        top = selected.get("top")
        middle = selected.get("middle")
        bottom = selected.get("bottom")
        item["pier_height"] = None
        item["embed_depth"] = None
        if valid_metric_triplet(top, middle, bottom, args):
            item["pier_height"] = round(top["value"] - middle["value"], 3)
            item["embed_depth"] = round(middle["value"] - bottom["value"], 3)
            if inferred_keys:
                selected["match_status"] = "rule_completed"
                item["status"] = f"rule_completed;inferred_{','.join(inferred_keys)}"
            else:
                selected["match_status"] = selected.get("match_status", "valid_global_constraint")
                item["status"] = selected["match_status"]
        else:
            if top and middle and top["value"] > middle["value"]:
                item["pier_height"] = round(top["value"] - middle["value"], 3)
            if middle and bottom and middle["value"] > bottom["value"] and bottom["value"] < 0:
                item["embed_depth"] = round(middle["value"] - bottom["value"], 3)
            selected["match_status"] = selected.get("match_status", "partial_constraint")
            item["status"] = selected["match_status"]
            if not (top and middle and top["value"] > middle["value"]):
                item["status"] += ";missing_or_invalid_top_middle"
            if not (middle and bottom and middle["value"] > bottom["value"] and bottom["value"] < 0):
                item["status"] += ";missing_or_invalid_middle_bottom"


def select_elevations(candidates: list[dict], pier_bbox: list[float]) -> dict:
    x1, y1, x2, y2 = pier_bbox
    center_x = (x1 + x2) / 2
    height = y2 - y1

    def score_for(target_y: float, value_hint: str, candidate: dict) -> float:
        cx, cy = candidate["center"]
        value = candidate["value"]
        distance_x = abs(cx - center_x)
        distance_y = abs(cy - target_y)
        score = distance_x * 0.45 + distance_y * 0.8 - candidate["confidence"] * 30
        if value_hint == "top" and value < 0:
            score += 1000
        if value_hint == "middle" and value < 0:
            score += 1000
        if value_hint == "bottom" and value > 0:
            score += 1000
        return score

    top_target = y1 + height * 0.08
    middle_target = y1 + height * 0.42
    bottom_target = y1 + height * 0.86

    positives = [c for c in candidates if c["value"] >= 0]
    negatives = [c for c in candidates if c["value"] < 0]

    top_pool = [c for c in positives if 13 <= c["value"] <= 22] or positives
    middle_pool = [c for c in positives if 3 <= c["value"] <= 13]
    bottom_pool = [c for c in negatives if -30 <= c["value"] <= 0] or negatives

    def same_candidate(a: dict | None, b: dict | None) -> bool:
        if not a or not b:
            return False
        return (
            abs(a["center"][0] - b["center"][0]) < 1e-6
            and abs(a["center"][1] - b["center"][1]) < 1e-6
            and abs(a["value"] - b["value"]) < 1e-6
        )

    min_role_y_gap = 12.0

    def y_order_ok(top: dict | None, middle: dict | None, bottom: dict | None) -> bool:
        if top and middle and not top["center"][1] + min_role_y_gap < middle["center"][1]:
            return False
        if middle and bottom and not middle["center"][1] + min_role_y_gap < bottom["center"][1]:
            return False
        if top and bottom and not top["center"][1] + min_role_y_gap < bottom["center"][1]:
            return False
        return True

    best_combo = None
    best_score = float("inf")
    for top_candidate in top_pool:
        for middle_candidate in middle_pool:
            if same_candidate(top_candidate, middle_candidate):
                continue
            if not y_order_ok(top_candidate, middle_candidate, None):
                continue
            if not top_candidate["value"] > middle_candidate["value"]:
                continue
            candidate_score = score_for(top_target, "top", top_candidate)
            candidate_score += score_for(middle_target, "middle", middle_candidate)
            # Prefer realistic pier heights, but don't hard-code one bridge.
            height = top_candidate["value"] - middle_candidate["value"]
            if not 3 <= height <= 16:
                candidate_score += 250
            if best_combo is None or candidate_score < best_score:
                best_combo = (top_candidate, middle_candidate)
                best_score = candidate_score

    if best_combo:
        top, middle = best_combo
    else:
        top = min(top_pool, key=lambda c: score_for(top_target, "top", c), default=None)
        remaining_positive = [
            c
            for c in positives
            if not same_candidate(c, top) and y_order_ok(top, c, None)
        ]
        middle = min(
            remaining_positive,
            key=lambda c: score_for(middle_target, "middle", c),
            default=None,
        )

    ordered_bottom_pool = [
        c
        for c in bottom_pool
        if not same_candidate(c, top)
        and not same_candidate(c, middle)
        and y_order_ok(top, middle, c)
    ]
    bottom = min(ordered_bottom_pool, key=lambda c: score_for(bottom_target, "bottom", c), default=None)

    # Fallback by value order when positional OCR is sparse.
    if top is None and positives:
        top = max(positives, key=lambda c: c["value"])
    if middle is None and positives:
        smaller = [
            c
            for c in positives
            if (top is None or c["value"] < top["value"])
            and not same_candidate(c, top)
            and y_order_ok(top, c, None)
        ]
        middle = max(smaller, key=lambda c: c["confidence"], default=None)
    if bottom is None and negatives:
        ordered_negatives = [
            c
            for c in negatives
            if not same_candidate(c, top)
            and not same_candidate(c, middle)
            and y_order_ok(top, middle, c)
        ]
        bottom = min(ordered_negatives, key=lambda c: c["value"], default=None)

    return {"top": top, "middle": middle, "bottom": bottom}


def select_local_elevations_with_status(candidates: list[dict], pier_bbox: list[float]) -> dict:
    selected = select_elevations(candidates, pier_bbox)
    selected["match_status"] = "local_geometry"
    selected["search_window"] = [pier_bbox[0], pier_bbox[2]]
    return selected


def best_global_role_candidate(
    global_candidates: list[dict],
    pier_bbox: list[float],
    role: str,
    args: argparse.Namespace,
) -> dict | None:
    x1, y1, x2, y2 = pier_bbox
    center_x = (x1 + x2) / 2
    role_target_y = {
        "top": y1 + (y2 - y1) * 0.06,
        "middle": y1 + (y2 - y1) * 0.36,
        "bottom": y1 + (y2 - y1) * 0.92,
    }[role]
    value_ranges = {
        "top": (args.top_min, args.top_max),
        "middle": (args.middle_min, args.middle_max),
        "bottom": (args.bottom_min, -1e-6),
    }
    low, high = value_ranges[role]
    max_dx = max(args.max_elevation_x_spread, (x2 - x1) * 0.75)
    max_dy = (y2 - y1) * (0.26 if role == "middle" else 0.36)
    candidates = [
        c
        for c in global_candidates
        if low <= c["value"] <= high
        and abs(c["center"][0] - center_x) <= max_dx
        and abs(c["center"][1] - role_target_y) <= max_dy
    ]
    if not candidates:
        return None

    def score(candidate: dict) -> float:
        cx, cy = candidate["center"]
        band_penalty = 0 if candidate.get("band_role") == role else 90
        correction_penalty = args.corrected_candidate_penalty if candidate.get("candidate_quality") == "corrected" else 0
        return (
            abs(cx - center_x) * args.x_weight
            + abs(cy - role_target_y) * (args.bottom_y_weight if role == "bottom" else args.band_y_weight)
            - candidate["confidence"] * 35
            + band_penalty
            + correction_penalty
        )

    return min(candidates, key=score)


def role_value_valid(candidate: dict | None, role: str, args: argparse.Namespace) -> bool:
    if not candidate:
        return False
    value = candidate["value"]
    if role == "top":
        return args.top_min <= value <= args.top_max
    if role == "middle":
        return args.middle_min <= value <= args.middle_max
    return args.bottom_min <= value < 0


def repair_local_selection_with_global(
    selected: dict,
    global_candidates: list[dict],
    pier_bbox: list[float],
    args: argparse.Namespace,
) -> dict:
    repaired = dict(selected)
    repaired_keys = []
    min_role_y_gap = getattr(args, "vertical_order_tolerance", 8.0)

    for role in ["top", "middle", "bottom"]:
        current = repaired.get(role)
        replacement = best_global_role_candidate(global_candidates, pier_bbox, role, args)
        if replacement is None:
            continue
        should_replace = False
        if not role_value_valid(current, role, args):
            should_replace = True
        elif role == "middle" and repaired.get("top") and current is repaired.get("top"):
            should_replace = True
        elif role == "bottom" and current and replacement["confidence"] > current.get("confidence", 0) + 0.12:
            # Keep local bottoms by default; only replace when global is clearly stronger.
            should_replace = True

        if should_replace:
            repaired[role] = replacement
            repaired_keys.append(role)

    top = repaired.get("top")
    middle = repaired.get("middle")
    bottom = repaired.get("bottom")
    if top and middle and not top["center"][1] + min_role_y_gap < middle["center"][1]:
        replacement = best_global_role_candidate(global_candidates, pier_bbox, "middle", args)
        if replacement and top["center"][1] + min_role_y_gap < replacement["center"][1] and top["value"] > replacement["value"]:
            repaired["middle"] = replacement
            repaired_keys.append("middle")
        else:
            repaired["middle"] = None
            repaired_keys.append("middle")
    middle = repaired.get("middle")
    if middle and bottom and not middle["center"][1] + min_role_y_gap < bottom["center"][1]:
        replacement = best_global_role_candidate(global_candidates, pier_bbox, "bottom", args)
        if replacement and middle["center"][1] + min_role_y_gap < replacement["center"][1] and middle["value"] > replacement["value"]:
            repaired["bottom"] = replacement
            repaired_keys.append("bottom")
        else:
            repaired["bottom"] = None
            repaired_keys.append("bottom")

    top = repaired.get("top")
    middle = repaired.get("middle")
    bottom = repaired.get("bottom")
    if top and middle and not top["value"] > middle["value"]:
        replacement = best_global_role_candidate(global_candidates, pier_bbox, "middle", args)
        if replacement and top["center"][1] + min_role_y_gap < replacement["center"][1] and top["value"] > replacement["value"]:
            repaired["middle"] = replacement
            repaired_keys.append("middle")
    if middle and bottom and not (middle["value"] > bottom["value"] and bottom["value"] < 0):
        replacement = best_global_role_candidate(global_candidates, pier_bbox, "bottom", args)
        if replacement and middle["center"][1] + min_role_y_gap < replacement["center"][1] and middle["value"] > replacement["value"]:
            repaired["bottom"] = replacement
            repaired_keys.append("bottom")

    if repaired_keys:
        repaired["match_status"] = "local_with_global_repair"
        repaired["global_repaired_keys"] = sorted(set(repaired_keys))
    return repaired


def detect_number_from_region(
    ocr_rows: list[dict],
    inferred_number: int,
    origin: tuple[int, int],
) -> dict:
    ox, oy = origin
    candidates = []
    for row in ocr_rows:
        number = parse_pier_number(row["text"])
        if number is None:
            continue
        x1, y1, x2, y2 = row["bbox"]
        candidates.append(
            {
                "number": number,
                "text": row["text"],
                "confidence": row["confidence"],
                "bbox": [x1 + ox, y1 + oy, x2 + ox, y2 + oy],
            }
        )

    if candidates:
        best = max(candidates, key=lambda item: item["confidence"])
        best["source"] = "ocr"
        return best

    return {
        "number": inferred_number,
        "text": None,
        "confidence": 0.0,
        "bbox": None,
        "source": "left_to_right",
    }


def draw_result(img: np.ndarray, item: dict) -> None:
    x1, y1, x2, y2 = [int(round(v)) for v in item["pier_bbox"]]
    color = (0, 180, 255)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

    number = item["pier_number"]["number"]
    height = item.get("pier_height")
    depth = item.get("embed_depth")
    label = f"#{number}"
    if height is not None and depth is not None:
        label += f" H={height:.3f} E={depth:.3f}"
    cv2.putText(img, label, (x1, max(25, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    for key, text_color in [("top", (0, 255, 0)), ("middle", (255, 0, 255)), ("bottom", (0, 0, 255))]:
        cand = item["selected_elevations"].get(key)
        if not cand:
            continue
        bx1, by1, bx2, by2 = [int(round(v)) for v in cand["bbox"]]
        cv2.rectangle(img, (bx1, by1), (bx2, by2), text_color, 2)
        cv2.putText(
            img,
            f"{key}:{cand['value']:.3f}",
            (bx1, max(25, by1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            text_color,
            2,
        )


def draw_lowest_water_level(img: np.ndarray, item: dict | None) -> None:
    if not item:
        return
    x1, y1, x2, y2 = [int(round(v)) for v in item["bbox"]]
    cx, cy = [int(round(v)) for v in item["center"]]
    color = (255, 180, 0)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    cv2.line(img, (max(0, cx - 320), cy), (min(img.shape[1] - 1, cx + 320), cy), color, 2)
    cv2.putText(
        img,
        f"LW={item['value']:.3f}",
        (x1, max(25, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2,
    )


def draw_bridge_metadata(img: np.ndarray, span_groups: list[dict], total_length: dict | None) -> None:
    span_color = (0, 220, 220)
    for item in span_groups:
        x1, y1, x2, y2 = [int(round(v)) for v in item["bbox"]]
        cv2.rectangle(img, (x1, y1), (x2, y2), span_color, 2)
        piers = item["pier_indices"]
        if piers:
            pier_text = f"P{piers[0]}-{piers[-1]}"
        else:
            pier_text = "P-"
        label = f"G{item['span_group_index']} {pier_text} {item['span_count']}x{item['span_length']}"
        cv2.putText(
            img,
            label,
            (x1, max(25, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            span_color,
            2,
        )

    if total_length:
        x1, y1, x2, y2 = [int(round(v)) for v in total_length["bbox"]]
        color = (80, 220, 255)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            img,
            f"TOTAL={total_length['value']}",
            (x1, max(25, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            color,
            2,
        )


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    debug_dir = output_dir / "debug_crops"
    output_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(str(args.image))
    if img is None:
        raise RuntimeError(f"Could not read image: {args.image}")
    image_height, image_width = img.shape[:2]

    detection_data = json.loads(Path(args.detections).read_text(encoding="utf-8"))
    pier_boxes = [
        det for det in detection_data["detections"] if det["class_name"] in set(args.pier_classes)
    ]
    pier_boxes.sort(key=lambda det: ((det["bbox"][0] + det["bbox"][2]) / 2, det["bbox"][1]))

    if args.max_piers > 0:
        pier_boxes = pier_boxes[: args.max_piers]

    print(f"Initializing PaddleOCR, piers: {len(pier_boxes)}")
    ocr = create_paddle_ocr(args)

    results = []
    annotated = img.copy()

    global_bbox = make_global_ocr_bbox(
        pier_boxes,
        image_width,
        image_height,
        pad_x=args.global_pad_x,
        pad_top=args.global_pad_top,
        pad_bottom=args.global_pad_bottom,
    )
    print(f"Running global multi-variant OCR in bbox: {global_bbox}")
    global_ocr_rows, global_candidates = run_global_multivariant_ocr(
        ocr,
        img,
        global_bbox,
        debug_dir,
        tile_width=args.global_tile_width,
        tile_overlap=args.global_tile_overlap,
    )
    elevation_bands = cluster_elevation_bands(global_candidates, y_tolerance=args.band_y_tolerance)
    lowest_water_level = extract_lowest_water_level(global_ocr_rows, (global_bbox[0], global_bbox[1]))
    span_groups = extract_span_groups(
        global_ocr_rows,
        (global_bbox[0], global_bbox[1]),
        args.start_number,
        len(pier_boxes),
    )
    total_length = extract_total_length(global_ocr_rows, (global_bbox[0], global_bbox[1]))
    print(f"Global OCR rows: {len(global_ocr_rows)}, elevation candidates: {len(global_candidates)}")
    print("Elevation bands:", elevation_bands)
    if lowest_water_level:
        print(
            f"Lowest water level: {lowest_water_level['value']:.3f} "
            f"from {lowest_water_level['text']} confidence={lowest_water_level['confidence']:.3f}"
        )
    if span_groups:
        print(
            "Span groups:",
            [
                {
                    "group": item["span_group_index"],
                    "text": item["text"],
                    "piers": item["pier_indices"],
                    "span_length": item["span_length"],
                }
                for item in span_groups
            ],
        )
    if total_length:
        print(
            f"Total length: {total_length['value']} "
            f"from {total_length['text']} confidence={total_length['confidence']:.3f}"
        )

    for index, det in enumerate(pier_boxes, start=args.start_number):
        bbox = det["bbox"]
        local_bbox = make_local_bbox(
            bbox,
            image_width,
            image_height,
            pad_x=args.local_pad_x,
            pad_top=args.local_pad_top,
            pad_bottom=args.local_pad_bottom,
        )
        number_bbox = make_number_bbox(
            bbox,
            image_width,
            image_height,
            pad_x=args.number_pad_x,
            down=args.number_down,
        )

        local_crop = crop_with_bbox(img, local_bbox)
        number_crop = crop_with_bbox(img, number_bbox)
        cv2.imwrite(str(debug_dir / f"pier_{index:02d}_local.png"), local_crop)
        cv2.imwrite(str(debug_dir / f"pier_{index:02d}_number.png"), number_crop)

        local_ocr = []
        if args.use_local_fallback:
            local_ocr = run_paddle_ocr(ocr, local_crop, scale=args.ocr_scale)
        number_ocr = run_paddle_ocr(ocr, number_crop, scale=args.number_ocr_scale)

        elevation_candidates = extract_elevation_candidates(local_ocr, (local_bbox[0], local_bbox[1]))
        if args.selection_mode == "local":
            selected = select_local_elevations_with_status(elevation_candidates, bbox)
        elif args.selection_mode == "local_global_repair":
            selected = select_local_elevations_with_status(elevation_candidates, bbox)
            selected = repair_local_selection_with_global(selected, global_candidates, bbox, args)
        else:
            selected = select_elevations_with_constraints(
                global_candidates,
                elevation_candidates,
                pier_boxes,
                index - args.start_number,
                args,
            )
        pier_number = detect_number_from_region(number_ocr, inferred_number=index, origin=(number_bbox[0], number_bbox[1]))

        top = selected.get("top")
        middle = selected.get("middle")
        bottom = selected.get("bottom")
        pier_height = None
        embed_depth = None
        status = selected.get("match_status", "ok")
        if top and middle and top["value"] > middle["value"]:
            pier_height = round(top["value"] - middle["value"], 3)
        else:
            status += ";missing_or_invalid_top_middle"
        if middle and bottom and middle["value"] > bottom["value"] and bottom["value"] < 0:
            embed_depth = round(middle["value"] - bottom["value"], 3)
        else:
            status += ";missing_or_invalid_middle_bottom"

        item = {
            "pier_index": index,
            "pier_bbox": bbox,
            "det_confidence": det["confidence"],
            "pier_number": pier_number,
            "selected_elevations": selected,
            "pier_height": pier_height,
            "embed_depth": embed_depth,
            "all_elevation_candidates": elevation_candidates,
            "global_candidates_in_window": [
                c
                for c in global_candidates
                if candidate_in_window(c, selected["search_window"][0], selected["search_window"][1])
            ],
            "search_window": selected["search_window"],
            "local_ocr": local_ocr,
            "number_ocr": number_ocr,
            "local_crop": str(debug_dir / f"pier_{index:02d}_local.png"),
            "number_crop": str(debug_dir / f"pier_{index:02d}_number.png"),
            "status": status,
        }
        for span_group in span_groups:
            if index in span_group["pier_indices"]:
                item["span_group"] = span_group["span_group_index"]
                item["span_length"] = span_group["span_length"]
                item["span_count_in_group"] = span_group["span_count"]
                break
        results.append(item)
        draw_result(annotated, item)
        print(
            f"Pier {index}: no={pier_number['number']} "
            f"top={top['value'] if top else None} mid={middle['value'] if middle else None} "
            f"bottom={bottom['value'] if bottom else None} H={pier_height} E={embed_depth} "
            f"status={status}"
        )

    postprocess_results(results, args)
    annotated = img.copy()
    for item in results:
        draw_result(annotated, item)
    draw_lowest_water_level(annotated, lowest_water_level)
    draw_bridge_metadata(annotated, span_groups, total_length)

    annotated_path = output_dir / "page_0019_pier_metrics.png"
    cv2.imwrite(str(annotated_path), annotated)

    json_path = output_dir / "page_0019_pier_metrics.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "image": str(args.image),
                "detections": str(args.detections),
                "start_number": args.start_number,
                "pier_count": len(results),
                "global_ocr_bbox": global_bbox,
                "global_ocr_rows": global_ocr_rows,
                "global_elevation_candidates": global_candidates,
                "elevation_bands": elevation_bands,
                "lowest_water_level": lowest_water_level,
                "span_groups": span_groups,
                "total_length": total_length,
                "results": results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Saved annotated image: {annotated_path}")
    print(f"Saved JSON: {json_path}")

    csv_path = output_dir / "page_0019_pier_metrics.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "pier_index",
                "pier_number",
                "number_source",
                "top_elevation",
                "middle_elevation",
                "bottom_elevation",
                "pier_height",
                "embed_depth",
                "span_group",
                "span_length",
                "match_status",
                "status",
            ]
        )
        for item in results:
            selected = item["selected_elevations"]
            writer.writerow(
                [
                    item["pier_index"],
                    item["pier_number"]["number"],
                    item["pier_number"]["source"],
                    fmt_value(selected["top"]["value"] if selected.get("top") else None),
                    fmt_value(selected["middle"]["value"] if selected.get("middle") else None),
                    fmt_value(selected["bottom"]["value"] if selected.get("bottom") else None),
                    fmt_value(item["pier_height"]),
                    fmt_value(item["embed_depth"]),
                    item.get("span_group", ""),
                    item.get("span_length", ""),
                    item["selected_elevations"].get("match_status", ""),
                    item["status"],
                ]
            )
    print(f"Saved CSV: {csv_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--detections", type=Path, default=DEFAULT_DETECTIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pier-classes", nargs="+", default=["C"])
    parser.add_argument("--start-number", type=int, default=1)
    parser.add_argument("--max-piers", type=int, default=0)
    parser.add_argument("--lang", default="ch")
    parser.add_argument("--text-detection-model-name", default="PP-OCRv4_server_det")
    parser.add_argument("--text-recognition-model-name", default="PP-OCRv4_server_rec")
    parser.add_argument("--use-gpu", action="store_true")
    parser.add_argument("--ocr-device", default="gpu:0")
    parser.add_argument("--stub-torch-import", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--local-pad-x", type=int, default=260)
    parser.add_argument("--local-pad-top", type=int, default=180)
    parser.add_argument("--local-pad-bottom", type=int, default=180)
    parser.add_argument("--number-pad-x", type=int, default=180)
    parser.add_argument("--number-down", type=int, default=320)
    parser.add_argument("--ocr-scale", type=float, default=2.0)
    parser.add_argument("--number-ocr-scale", type=float, default=3.0)
    parser.add_argument("--global-pad-x", type=int, default=360)
    parser.add_argument("--global-pad-top", type=int, default=260)
    parser.add_argument("--global-pad-bottom", type=int, default=260)
    parser.add_argument("--global-tile-width", type=int, default=2200)
    parser.add_argument("--global-tile-overlap", type=int, default=360)
    parser.add_argument("--band-y-tolerance", type=float, default=85.0)
    parser.add_argument("--global-match-margin", type=float, default=170.0)
    parser.add_argument("--x-weight", type=float, default=0.28)
    parser.add_argument("--band-y-weight", type=float, default=0.16)
    parser.add_argument("--bottom-y-weight", type=float, default=0.32)
    parser.add_argument("--bottom-value-weight", type=float, default=1.5)
    parser.add_argument("--corrected-candidate-penalty", type=float, default=45.0)
    parser.add_argument("--max-elevation-x-spread", type=float, default=420.0)
    parser.add_argument("--alignment-x-weight", type=float, default=0.55)
    parser.add_argument("--vertical-order-tolerance", type=float, default=8.0)
    parser.add_argument("--vertical-spacing-weight", type=float, default=0.18)
    parser.add_argument("--require-geometry", action="store_true")
    parser.add_argument(
        "--selection-mode",
        choices=["global", "local", "local_global_repair"],
        default="local_global_repair",
    )
    parser.add_argument("--use-local-fallback", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--top-min", type=float, default=13.0)
    parser.add_argument("--top-max", type=float, default=22.0)
    parser.add_argument("--middle-min", type=float, default=3.0)
    parser.add_argument("--middle-max", type=float, default=13.0)
    parser.add_argument("--bottom-min", type=float, default=-35.0)
    parser.add_argument("--min-pier-height", type=float, default=3.0)
    parser.add_argument("--max-pier-height", type=float, default=16.0)
    parser.add_argument("--min-embed-depth", type=float, default=8.0)
    parser.add_argument("--max-embed-depth", type=float, default=40.0)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

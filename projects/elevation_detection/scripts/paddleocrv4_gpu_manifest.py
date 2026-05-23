"""
Run PaddleOCRv4 on image crops listed in a JSON manifest.

This helper intentionally imports no torch/ultralytics modules so Paddle can
initialize CUDA cleanly on Windows.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

CONDA_DLL_DIR = Path(sys.prefix) / "Library" / "bin"
if os.name == "nt" and CONDA_DLL_DIR.exists():
    os.environ["PATH"] = f"{CONDA_DLL_DIR}{os.pathsep}{os.environ.get('PATH', '')}"
    os.add_dll_directory(str(CONDA_DLL_DIR))

NVIDIA_DLL_ROOT = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
if os.name == "nt" and NVIDIA_DLL_ROOT.exists():
    for nvidia_bin in NVIDIA_DLL_ROOT.rglob("bin"):
        os.environ["PATH"] = f"{nvidia_bin}{os.pathsep}{os.environ.get('PATH', '')}"
        os.add_dll_directory(str(nvidia_bin))

import cv2
import numpy as np


NUMERIC_TOKEN_RE = re.compile(r"[-+]?\d+(?:[.,]\d+)?")


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
    """Avoid optional modelscope->torch import during PaddleOCR initialization."""
    if "torch" in sys.modules:
        return

    import importlib.machinery
    import types

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


def numeric_text(rows: list[dict]) -> str | None:
    joined = " ".join(row["text"] for row in rows)
    matches = NUMERIC_TOKEN_RE.findall(joined.replace("O", "0").replace("o", "0"))
    if not matches:
        return None
    return max(matches, key=len).replace(",", ".")


def create_ocr(args: argparse.Namespace) -> Any:
    import paddle

    if not paddle.device.is_compiled_with_cuda():
        raise RuntimeError("Paddle is not compiled with CUDA.")
    paddle.device.set_device(args.device)
    print(f"Paddle device: {paddle.device.get_device()}")

    install_torch_import_stub()
    from paddleocr import PaddleOCR

    return PaddleOCR(
        lang=args.lang,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        text_detection_model_name=args.text_detection_model_name,
        text_recognition_model_name=args.text_recognition_model_name,
        device=args.device,
    )


def run_ocr(ocr: Any, image: np.ndarray, scale: float) -> list[dict]:
    if image.size == 0:
        return []
    if scale != 1.0:
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    result = ocr.predict(image)
    rows: list[dict] = []
    if not result:
        return rows
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
        rows.append(
            {
                "text": str(text),
                "confidence": float(score),
                "bbox": [float(x1 / scale), float(y1 / scale), float(x2 / scale), float(y2 / scale)],
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="gpu:0")
    parser.add_argument("--lang", default="ch")
    parser.add_argument("--ocr-scale", type=float, default=3.0)
    parser.add_argument("--text-detection-model-name", default="PP-OCRv4_server_det")
    parser.add_argument("--text-recognition-model-name", default="PP-OCRv4_server_rec")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    ocr = create_ocr(args)
    results = []
    for item in manifest["items"]:
        image = cv2.imread(item["ocr_crop"])
        if image is None:
            raise RuntimeError(f"Could not read OCR crop: {item['ocr_crop']}")
        rows = run_ocr(ocr, image, args.ocr_scale)
        results.append(
            {
                **item,
                "ocr_engine": "paddleocrv4",
                "ocr_device": args.device,
                "ocr_rows": rows,
                "ocr_text": " ".join(row["text"] for row in rows),
                "numeric_value_text": numeric_text(rows),
            }
        )
        print(
            f"{item['class_name']} conf={item['confidence']:.3f} "
            f"value={results[-1]['numeric_value_text']} text={results[-1]['ocr_text']!r}"
        )
    args.output.write_text(json.dumps({"items": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved OCR JSON: {args.output}")


if __name__ == "__main__":
    main()

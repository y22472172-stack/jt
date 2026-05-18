"""
PDF scanned document to enhanced images converter.
Optimized for 1998-era scanned engineering drawings with fine line details.
"""

import os
import sys
import math
import numpy as np
import cv2
import fitz  # PyMuPDF


def estimate_skew(gray: np.ndarray) -> float:
    """Estimate skew angle using projection profile method."""
    # binarize
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # crop center 80% to avoid border noise
    h, w = binary.shape
    margin_h, margin_w = int(h * 0.1), int(w * 0.1)
    roi = binary[margin_h:h - margin_h, margin_w:w - margin_w]

    best_angle = 0.0
    best_score = 0
    # search -3 to +3 degrees in 0.1 steps
    for angle_10x in range(-30, 31):
        angle = angle_10x / 10.0
        M = cv2.getRotationMatrix2D((roi.shape[1] // 2, roi.shape[0] // 2), angle, 1.0)
        rotated = cv2.warpAffine(roi, M, (roi.shape[1], roi.shape[0]),
                                 flags=cv2.INTER_NEAREST, borderValue=0)
        # horizontal projection profile
        proj = np.sum(rotated, axis=1).astype(np.float64)
        # score = sum of squared differences (sharper peak = better alignment)
        score = np.sum(np.diff(proj) ** 2)
        if score > best_score:
            best_score = score
            best_angle = angle
    return best_angle


def deskew_image(img: np.ndarray, angle: float) -> np.ndarray:
    """Rotate image to correct skew."""
    if abs(angle) < 0.05:
        return img
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    # compute new bounding box
    cos_a = abs(M[0, 0])
    sin_a = abs(M[0, 1])
    new_w = int(h * sin_a + w * cos_a)
    new_h = int(h * cos_a + w * sin_a)
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2
    border = 255 if len(img.shape) == 2 else (255, 255, 255)
    return cv2.warpAffine(img, M, (new_w, new_h), borderValue=border)


def enhance_scanned_page(img_bgr: np.ndarray, do_deskew: bool = True,
                         denoise_strength: int = 7) -> np.ndarray:
    """
    Full enhancement pipeline for a scanned engineering drawing page.
    """
    # 0. convert to grayscale for analysis
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 1. de-skew
    if do_deskew:
        skew_angle = estimate_skew(gray)
        if abs(skew_angle) > 0.1:
            img_bgr = deskew_image(img_bgr, skew_angle)
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 2. denoise (bilateral preserves edges, good for line drawings)
    if denoise_strength > 0:
        gray = cv2.bilateralFilter(gray, denoise_strength, 50, 50)

    # 3. CLAHE (adaptive histogram equalization) — best for uneven lighting in scans
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)

    # 4. adaptive sharpening via unsharp mask
    blurred = cv2.GaussianBlur(enhanced_gray, (0, 0), 3)
    sharpened = cv2.addWeighted(enhanced_gray, 1.8, blurred, -0.8, 0)
    # clamp
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

    # 5. mild morphological cleanup: close small gaps in lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    sharpened = cv2.morphologyEx(sharpened, cv2.MORPH_CLOSE, kernel)

    # 6. convert back to BGR for output
    result = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)

    return result


def convert_pdf_to_images(
    pdf_path: str,
    output_dir: str = None,
    dpi: int = 400,
    do_deskew: bool = True,
    denoise_strength: int = 7,
):
    """
    Convert scanned PDF pages to enhanced images.
    """
    if not os.path.isfile(pdf_path):
        print(f"Error: File not found: {pdf_path}")
        sys.exit(1)

    if output_dir is None:
        base = os.path.splitext(os.path.basename(pdf_path))[0]
        output_dir = os.path.join(os.path.dirname(pdf_path), f"{base}_images")

    os.makedirs(output_dir, exist_ok=True)

    doc = fitz.open(pdf_path)
    total = len(doc)
    print(f"PDF: {pdf_path}")
    print(f"Total pages: {total}")
    print(f"DPI: {dpi} | Deskew: {do_deskew} | Denoise: {denoise_strength}")
    print(f"Output: {output_dir}")
    print("-" * 60)

    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)

    for page_num in range(total):
        page = doc[page_num]
        page_label = f"page_{page_num + 1:04d}"

        # render at high DPI
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        # convert to numpy BGR (OpenCV format)
        img_rgb = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        # enhance
        enhanced = enhance_scanned_page(img_bgr, do_deskew=do_deskew,
                                        denoise_strength=denoise_strength)

        # save as high-quality PNG
        out_path = os.path.join(output_dir, f"{page_label}.png")
        cv2.imwrite(out_path, enhanced, [cv2.IMWRITE_PNG_COMPRESSION, 3])

        pct = (page_num + 1) / total * 100
        print(f"  [{page_num + 1:4d}/{total}] {pct:5.1f}%  {out_path}")

    doc.close()
    print("-" * 60)
    print(f"Done. {total} enhanced images saved to: {output_dir}")
    return output_dir


if __name__ == "__main__":
    pdf = os.path.join(os.path.dirname(os.path.abspath(__file__)), "GL5.102.AZ-507(1).pdf")
    convert_pdf_to_images(pdf, dpi=400, do_deskew=True, denoise_strength=7)

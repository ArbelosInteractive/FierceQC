
import json
import math
import os
import re
import uuid
import difflib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
try:
    from skimage.color import rgb2lab  # type: ignore
except Exception:  # pragma: no cover
    rgb2lab = None

try:
    from skimage.feature import hog, local_binary_pattern  # type: ignore
except Exception:  # pragma: no cover
    hog = None
    local_binary_pattern = None

try:
    from skimage.metrics import structural_similarity as ssim  # type: ignore
except Exception:  # pragma: no cover
    ssim = None

try:
    import pytesseract
except Exception:  # pragma: no cover
    pytesseract = None

try:
    import face_recognition  # type: ignore
except Exception:  # pragma: no cover
    face_recognition = None

try:
    import folder_paths  # type: ignore
except Exception:  # pragma: no cover
    folder_paths = None


PACKAGE_CATEGORY = "Arbelos/VTON QC"
EPS = 1e-8
CANONICAL_GARMENT_LABELS = {
    'hat': ['hat', 'cap', 'cowboy hat', 'beanie', 'fedora'],
    'shirt': ['shirt', 'top', 'blouse', 'tee', 'tshirt', 't-shirt', 'jacket'],
    'pants': ['pants', 'jeans', 'trousers', 'leggings'],
    'skirt': ['skirt'],
    'dress': ['dress', 'gown'],
    'shoes': ['shoes', 'shoe', 'boots', 'sneakers', 'heels', 'sandals'],
    'purse': ['purse', 'bag', 'handbag', 'tote'],
}

LOCAL_OCR_FONTS = [
    cv2.FONT_HERSHEY_SIMPLEX,
    cv2.FONT_HERSHEY_DUPLEX,
    cv2.FONT_HERSHEY_COMPLEX,
    cv2.FONT_HERSHEY_TRIPLEX,
]

KNOWN_LABEL_VARIANTS = sorted({variant for variants in CANONICAL_GARMENT_LABELS.values() for variant in variants} | set(CANONICAL_GARMENT_LABELS.keys()), key=len)


def _rgb2lab_image(image: np.ndarray) -> np.ndarray:
    if rgb2lab is not None:
        return rgb2lab(image.astype(np.float32) / 255.0)
    lab = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    # approximate skimage scale: L in 0..100, a/b around -128..127
    lab[..., 0] = lab[..., 0] * (100.0 / 255.0)
    lab[..., 1:] = lab[..., 1:] - 128.0
    return lab


def _hog_feature(gray: np.ndarray) -> np.ndarray:
    gray = np.asarray(gray, dtype=np.uint8)
    if hog is not None:
        return hog(gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=False)
    bins = np.linspace(0.0, 2.0 * np.pi, 10, endpoint=True)
    hist, _ = np.histogram(ang, bins=bins, weights=mag)
    hist = hist.astype(np.float32)
    norm = float(np.linalg.norm(hist)) + EPS
    return hist / norm


def _lbp_histogram(gray: np.ndarray, points: int = 24, radius: int = 3) -> np.ndarray:
    gray = np.asarray(gray, dtype=np.uint8)
    if local_binary_pattern is not None:
        lbp = local_binary_pattern(gray, points, radius, method="uniform")
        bins = int(points + 2)
        hist, _ = np.histogram(lbp.ravel(), bins=bins, range=(0, bins), density=True)
        return hist.astype(np.float32)
    hist = cv2.calcHist([gray], [0], None, [32], [0, 256]).reshape(-1).astype(np.float32)
    hist /= float(hist.sum()) + EPS
    return hist


def _ssim_score(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.uint8)
    b = np.asarray(b, dtype=np.uint8)
    if ssim is not None:
        return float(ssim(a, b, data_range=255))
    diff = a.astype(np.float32) - b.astype(np.float32)
    mse = float(np.mean(diff * diff))
    return float(max(-1.0, min(1.0, 1.0 - mse / (255.0 * 255.0))))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b)) + EPS
    return _clamp01((float(np.dot(a, b)) / denom + 1.0) / 2.0)


def _json_dumps(data: Any) -> str:
    return json.dumps(data, indent=2, sort_keys=False)


def _get_output_dir() -> str:
    if folder_paths is not None and hasattr(folder_paths, "get_output_directory"):
        try:
            return folder_paths.get_output_directory()
        except Exception:
            pass
    default_dir = os.path.join(os.getcwd(), "output")
    os.makedirs(default_dir, exist_ok=True)
    return default_dir


def _tensor_to_np(image: torch.Tensor) -> np.ndarray:
    """
    ComfyUI IMAGE -> numpy uint8 RGB.
    IMAGE is [B, H, W, C] float 0..1; single image treated as batch size 1.
    """
    if not isinstance(image, torch.Tensor):
        image = torch.tensor(image)
    arr = image.detach().cpu().float().numpy()
    if arr.ndim == 4:
        arr = arr[0]
    arr = np.clip(arr, 0.0, 1.0)
    arr = (arr * 255.0).round().astype(np.uint8)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    return arr


def _np_to_tensor(image: np.ndarray) -> torch.Tensor:
    arr = np.asarray(image)
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32) / 255.0
    if arr.ndim == 3:
        arr = arr[None, ...]
    return torch.from_numpy(arr)


def _resize_rgb(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def _rgb_to_gray(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def _softmax(values: Sequence[float]) -> List[float]:
    if not values:
        return []
    arr = np.asarray(values, dtype=np.float32)
    arr = arr - np.max(arr)
    exp = np.exp(arr)
    denom = float(np.sum(exp)) + EPS
    return (exp / denom).tolist()


def _normalize_enabled_weights(section_map: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
    enabled = []
    for key, section in section_map.items():
        if bool(section.get("enabled", True)) and _safe_float(section.get("weight", 0.0), 0.0) > 0:
            enabled.append((key, _safe_float(section.get("weight", 0.0), 0.0)))
    total = sum(weight for _, weight in enabled)
    if total <= 0:
        return {key: 0.0 for key in section_map.keys()}
    return {key: (weight / total if any(k == key for k, _ in enabled) else 0.0) for key, weight in [(k, section_map[k].get("weight", 0.0)) for k in section_map.keys()]}


def _normalize_subweights(subsections: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
    enabled = {k: _safe_float(v.get("weight", 0.0), 0.0) for k, v in subsections.items() if bool(v.get("enabled", True)) and _safe_float(v.get("weight", 0.0), 0.0) > 0}
    total = sum(enabled.values())
    out: Dict[str, float] = {}
    for key in subsections.keys():
        out[key] = enabled.get(key, 0.0) / total if total > 0 else 0.0
    return out


def _largest_connected_component(mask: np.ndarray) -> np.ndarray:
    mask_u8 = (mask > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if num_labels <= 1:
        return mask_u8
    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == best).astype(np.uint8)


def _bbox_from_mask(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    return x0, y0, x1, y1


def _crop(image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = bbox
    h, w = image.shape[:2]
    x0 = max(0, min(w - 1, x0))
    x1 = max(x0 + 1, min(w, x1))
    y0 = max(0, min(h - 1, y0))
    y1 = max(y0 + 1, min(h, y1))
    return image[y0:y1, x0:x1].copy()


def _expand_bbox(bbox: Tuple[int, int, int, int], scale: float, image_shape: Tuple[int, int, int]) -> Tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox
    h, w = image_shape[:2]
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    bw = (x1 - x0) * scale
    bh = (y1 - y0) * scale
    nx0 = int(max(0, math.floor(cx - bw / 2.0)))
    ny0 = int(max(0, math.floor(cy - bh / 2.0)))
    nx1 = int(min(w, math.ceil(cx + bw / 2.0)))
    ny1 = int(min(h, math.ceil(cy + bh / 2.0)))
    return nx0, ny0, nx1, ny1


def _white_bg_foreground_mask(rgb: np.ndarray, threshold: int = 245) -> np.ndarray:
    gray = _rgb_to_gray(rgb)
    mask = (gray < threshold).astype(np.uint8)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def _segment_person(rgb: np.ndarray) -> Tuple[np.ndarray, float]:
    h, w = rgb.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    rect_margin_x = max(5, int(w * 0.1))
    rect_margin_y = max(5, int(h * 0.05))
    rect = (rect_margin_x, rect_margin_y, max(1, w - 2 * rect_margin_x), max(1, h - 2 * rect_margin_y))
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), mask, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
        fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)
    except Exception:
        fg = _white_bg_foreground_mask(rgb)
    fg = _largest_connected_component(fg)
    bbox = _bbox_from_mask(fg)
    if bbox is None:
        fg = np.ones((h, w), np.uint8)
        confidence = 0.15
    else:
        area_ratio = float(fg.sum()) / float(h * w + EPS)
        confidence = _clamp01(1.0 - abs(area_ratio - 0.35) / 0.35)
    return fg, confidence


def _extract_face_bbox(rgb: np.ndarray) -> Tuple[Tuple[int, int, int, int], float]:
    h, w = rgb.shape[:2]
    if face_recognition is not None:
        try:
            boxes = face_recognition.face_locations(rgb)
            if boxes:
                top, right, bottom, left = boxes[0]
                bbox = (left, top, right, bottom)
                area = max(1, (right - left) * (bottom - top))
                conf = _clamp01(min(1.0, area / float(h * w * 0.12)))
                return bbox, max(conf, 0.65)
        except Exception:
            pass

    gray = _rgb_to_gray(rgb)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    rects = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(max(24, w // 16), max(24, h // 16)))
    if len(rects) > 0:
        x, y, fw, fh = max(rects, key=lambda r: r[2] * r[3])
        bbox = (int(x), int(y), int(x + fw), int(y + fh))
        area = max(1, fw * fh)
        conf = _clamp01(min(1.0, area / float(h * w * 0.08)))
        return bbox, max(conf, 0.55)

    # fallback upper-center crop
    fw = max(24, int(w * 0.28))
    fh = max(24, int(h * 0.22))
    x0 = max(0, (w - fw) // 2)
    y0 = max(0, int(h * 0.08))
    bbox = (x0, y0, min(w, x0 + fw), min(h, y0 + fh))
    return bbox, 0.2


def _lbp_hist(gray: np.ndarray, points: int = 24, radius: int = 3) -> np.ndarray:
    return _lbp_histogram(gray, points=points, radius=radius)


def _face_feature_vector(rgb: np.ndarray) -> np.ndarray:
    bbox, _ = _extract_face_bbox(rgb)
    crop = _crop(rgb, _expand_bbox(bbox, 1.35, rgb.shape))
    crop = _resize_rgb(crop, (112, 112))
    gray = _rgb_to_gray(crop)
    if face_recognition is not None:
        try:
            enc = face_recognition.face_encodings(crop)
            if enc:
                return np.asarray(enc[0], dtype=np.float32)
        except Exception:
            pass
    hog_feat = _hog_feature(gray)
    lbp_feat = _lbp_hist(gray)
    mean_rgb = crop.reshape(-1, 3).mean(axis=0) / 255.0
    return np.concatenate([hog_feat.astype(np.float32), lbp_feat, mean_rgb.astype(np.float32)], axis=0)


def _face_geometry_score(src_rgb: np.ndarray, dst_rgb: np.ndarray) -> float:
    src_bbox, _ = _extract_face_bbox(src_rgb)
    dst_bbox, _ = _extract_face_bbox(dst_rgb)
    sx0, sy0, sx1, sy1 = src_bbox
    dx0, dy0, dx1, dy1 = dst_bbox
    src_ratio = (sx1 - sx0) / max(1.0, (sy1 - sy0))
    dst_ratio = (dx1 - dx0) / max(1.0, (dy1 - dy0))
    ratio_delta = abs(src_ratio - dst_ratio)
    return _clamp01(1.0 - ratio_delta / 1.25)


def _blur_score(rgb: np.ndarray) -> float:
    gray = _rgb_to_gray(rgb)
    val = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return _clamp01(val / 300.0)


def _visibility_confidence(mask: Optional[np.ndarray], bbox: Optional[Tuple[int, int, int, int]], image_shape: Tuple[int, int, int], extra_quality: float = 1.0) -> float:
    h, w = image_shape[:2]
    confs: List[float] = []
    if mask is not None:
        area_ratio = float(mask.sum()) / float(h * w + EPS)
        confs.append(_clamp01(min(1.0, area_ratio / 0.2)))
    if bbox is not None:
        x0, y0, x1, y1 = bbox
        area = max(1, (x1 - x0) * (y1 - y0))
        confs.append(_clamp01(min(1.0, area / float(h * w * 0.2))))
        edge_margin = min(x0, y0, w - x1, h - y1)
        confs.append(_clamp01((edge_margin + 1) / max(1.0, 0.1 * min(h, w))))
    confs.append(_clamp01(extra_quality))
    return float(np.mean(confs)) if confs else 0.5


def _hist_similarity_rgb(a: np.ndarray, b: np.ndarray, bins: int = 16) -> float:
    sims = []
    for ch in range(3):
        hist_a = cv2.calcHist([a], [ch], None, [bins], [0, 256]).reshape(-1)
        hist_b = cv2.calcHist([b], [ch], None, [bins], [0, 256]).reshape(-1)
        hist_a /= float(hist_a.sum()) + EPS
        hist_b /= float(hist_b.sum()) + EPS
        sims.append(_clamp01(cv2.compareHist(hist_a.astype(np.float32), hist_b.astype(np.float32), cv2.HISTCMP_CORREL)))
    return float(np.mean(sims))


def _lab_color_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_lab = _rgb2lab_image(a)
    b_lab = _rgb2lab_image(b)
    mean_a = a_lab.reshape(-1, 3).mean(axis=0)
    mean_b = b_lab.reshape(-1, 3).mean(axis=0)
    dist = np.linalg.norm(mean_a - mean_b)
    return _clamp01(1.0 - dist / 60.0)


def _texture_similarity(a: np.ndarray, b: np.ndarray) -> float:
    ga = _rgb_to_gray(_resize_rgb(a, (128, 128)))
    gb = _rgb_to_gray(_resize_rgb(b, (128, 128)))
    lbp_a = _lbp_hist(ga)
    lbp_b = _lbp_hist(gb)
    ssim_val = _ssim_score(ga, gb)
    return _clamp01(0.55 * _cosine_similarity(lbp_a, lbp_b) + 0.45 * ((ssim_val + 1.0) / 2.0))


def _edge_similarity(a: np.ndarray, b: np.ndarray) -> float:
    ga = _rgb_to_gray(_resize_rgb(a, (128, 128)))
    gb = _rgb_to_gray(_resize_rgb(b, (128, 128)))
    ea = cv2.Canny(ga, 60, 160)
    eb = cv2.Canny(gb, 60, 160)
    inter = float(np.logical_and(ea > 0, eb > 0).sum())
    union = float(np.logical_or(ea > 0, eb > 0).sum()) + EPS
    iou = inter / union
    density_sim = 1.0 - abs(float((ea > 0).mean()) - float((eb > 0).mean()))
    return _clamp01(0.65 * iou + 0.35 * density_sim)


def _normalize_label_text(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9]+", " ", (text or "")).strip().lower()
    if not text:
        return ""
    candidates = []
    for canonical, variants in CANONICAL_GARMENT_LABELS.items():
        if text == canonical:
            return canonical
        if any(v in text for v in variants):
            return canonical
        candidates.extend([canonical] + variants)
    match = difflib.get_close_matches(text, candidates, n=1, cutoff=0.55)
    if match:
        best = match[0]
        for canonical, variants in CANONICAL_GARMENT_LABELS.items():
            if best == canonical or best in variants:
                return canonical
    return text


def _extract_label_tokens(text: str) -> List[str]:
    text = _normalize_label_text(text)
    if not text:
        return []
    out: List[str] = []
    for canonical, variants in CANONICAL_GARMENT_LABELS.items():
        if text == canonical or any(v in text for v in variants):
            out.append(canonical)
    if not out and text:
        out.append(text)
    return out




def _prepare_label_ocr_image(rgb: np.ndarray) -> np.ndarray:
    gray = _rgb_to_gray(rgb)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    # text is usually dark on a bright label box
    bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    # trim edges and resize to a stable height
    ys, xs = np.where(bw > 0)
    if len(xs) and len(ys):
        x0, x1 = max(0, int(xs.min()) - 2), min(bw.shape[1], int(xs.max()) + 3)
        y0, y1 = max(0, int(ys.min()) - 2), min(bw.shape[0], int(ys.max()) + 3)
        bw = bw[y0:y1, x0:x1]
    if bw.shape[0] < 18:
        scale = 18.0 / max(1.0, float(bw.shape[0]))
        bw = cv2.resize(bw, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return bw


def _render_text_template(word: str, width: int, height: int) -> np.ndarray:
    best = None
    word = word.lower().strip()
    for font in LOCAL_OCR_FONTS:
        for thickness in (1, 2):
            canvas = np.zeros((height, width), np.uint8)
            # search a font scale that fills the canvas without clipping
            scale = 0.2
            best_scale = scale
            while scale <= 3.0:
                (tw, th), baseline = cv2.getTextSize(word, font, scale, thickness)
                if tw <= width - 4 and th + baseline <= height - 4:
                    best_scale = scale
                    scale += 0.05
                else:
                    break
            (tw, th), baseline = cv2.getTextSize(word, font, best_scale, thickness)
            x = max(1, (width - tw) // 2)
            y = max(th + 1, (height + th) // 2 - baseline // 2)
            cv2.putText(canvas, word, (x, y), font, best_scale, 255, thickness, cv2.LINE_AA)
            score = float(canvas.mean())
            if best is None or score > best[0]:
                best = (score, canvas)
    return best[1] if best is not None else np.zeros((height, width), np.uint8)


def _local_known_label_ocr(rgb: np.ndarray) -> str:
    proc = _prepare_label_ocr_image(rgb)
    h, w = proc.shape[:2]
    if h < 6 or w < 6:
        return ""
    best_word = ""
    best_score = -1.0
    for word in KNOWN_LABEL_VARIANTS:
        tpl = _render_text_template(word, w, h)
        # use both IoU and correlation to tolerate font mismatch
        proc_bin = (proc > 0).astype(np.uint8)
        tpl_bin = (tpl > 0).astype(np.uint8)
        inter = float(np.logical_and(proc_bin > 0, tpl_bin > 0).sum())
        union = float(np.logical_or(proc_bin > 0, tpl_bin > 0).sum()) + EPS
        iou = inter / union
        corr = cv2.matchTemplate(proc.astype(np.float32), tpl.astype(np.float32), cv2.TM_CCOEFF_NORMED)[0, 0]
        # length prior helps prefer exact small words like hat over longer words
        prior = 1.0 - (abs(len(word) - max(1, w // max(6, h))) * 0.02)
        score = 0.55 * float(max(-1.0, min(1.0, corr))) + 0.45 * iou + 0.05 * prior
        if score > best_score:
            best_score = score
            best_word = word
    return best_word if best_score >= 0.28 else ""


def _ocr_text(rgb: np.ndarray, psm: int = 6) -> str:
    # First try bundled local OCR restricted to known garment labels so the node
    # works even when the external Tesseract binary is unavailable.
    local = _local_known_label_ocr(rgb)
    if local:
        return local
    if pytesseract is None:
        return ""
    try:
        gray = _rgb_to_gray(rgb)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        config = f"--psm {psm} -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789- "
        text = pytesseract.image_to_string(gray, config=config)
        text = re.sub(r"[^A-Za-z0-9]+", " ", text).strip().lower()
        return text
    except Exception:
        return ""


def _text_similarity(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    set_a = set(a.split())
    set_b = set(b.split())
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return _clamp01(inter / max(1.0, union))


def _connected_components_boxes(mask: np.ndarray, min_area: int = 800) -> List[Tuple[int, int, int, int]]:
    mask_u8 = (mask > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    boxes = []
    for idx in range(1, num_labels):
        x, y, w, h, area = stats[idx]
        if area >= min_area:
            boxes.append((int(x), int(y), int(x + w), int(y + h)))
    return boxes


def _estimate_border_bg_color(rgb: np.ndarray, border: int = 8) -> np.ndarray:
    h, w = rgb.shape[:2]
    border = max(1, min(border, h // 4 if h > 4 else 1, w // 4 if w > 4 else 1))
    strips = [rgb[:border, :, :], rgb[-border:, :, :], rgb[:, :border, :], rgb[:, -border:, :]]
    pixels = np.concatenate([s.reshape(-1, 3) for s in strips], axis=0)
    return np.median(pixels, axis=0).astype(np.float32)


def _background_distance_mask(rgb: np.ndarray, threshold: float = 26.0) -> np.ndarray:
    bg = _estimate_border_bg_color(rgb)
    dist = np.linalg.norm(rgb.astype(np.float32) - bg[None, None, :], axis=2)
    mask = (dist >= threshold).astype(np.uint8)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def _mask_boxes(mask: np.ndarray, boxes: Sequence[Tuple[int, int, int, int]], value: int = 0) -> np.ndarray:
    out = mask.copy()
    for x0, y0, x1, y1 in boxes:
        out[y0:y1, x0:x1] = value
    return out


def _detect_label_boxes(collage_rgb: np.ndarray) -> List[Tuple[int, int, int, int]]:
    gray = _rgb_to_gray(collage_rgb)
    bright = (gray > 242).astype(np.uint8)
    bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    boxes = []
    h, w = gray.shape[:2]
    for x0, y0, x1, y1 in _connected_components_boxes(bright, min_area=max(40, (h * w) // 40000)):
        bw = x1 - x0
        bh = y1 - y0
        area = bw * bh
        if bw < 12 or bh < 8:
            continue
        if bw > int(w * 0.35) or bh > int(h * 0.15):
            continue
        border_near = (x0 < w * 0.2 or y0 < h * 0.2 or x1 > w * 0.8 or y1 > h * 0.8)
        crop = collage_rgb[y0:y1, x0:x1]
        dark_ratio = float((_rgb_to_gray(crop) < 120).mean())
        fill_ratio = area / max(1.0, h * w)
        if border_near and dark_ratio > 0.01 and fill_ratio < 0.05:
            pad = 3
            boxes.append((max(0, x0 - pad), max(0, y0 - pad), min(w, x1 + pad), min(h, y1 + pad)))
    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
    return boxes


def _parse_collage_labels(collage_rgb: np.ndarray) -> List[Dict[str, Any]]:
    boxes = _detect_label_boxes(collage_rgb)
    labels: List[Dict[str, Any]] = []
    h, w = collage_rgb.shape[:2]
    for box in boxes:
        crop = _crop(collage_rgb, box)
        gray = _rgb_to_gray(crop)
        variants = []
        variants.append((gray, 8))
        variants.append((cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1], 8))
        enlarged = cv2.resize(gray, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
        variants.append((enlarged, 8))
        texts = []
        if pytesseract is not None:
            for proc, psm in variants:
                try:
                    txt = pytesseract.image_to_string(proc, config=f"--psm {psm} -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz").strip().lower()
                except Exception:
                    txt = ""
                if txt:
                    texts.append(txt)
        if not texts:
            text = _ocr_text(crop, psm=8)
            texts = [text] if text else []
        chosen = ""
        tokens: List[str] = []
        for txt in texts:
            curr = _extract_label_tokens(txt)
            if curr:
                chosen = txt
                tokens = curr
                break
        if not tokens:
            continue
        labels.append({
            'box': box,
            'text': chosen,
            'canonical_labels': tokens,
            'primary_label': tokens[0],
            'anchor': ((box[0] + box[2]) / 2.0 / max(1.0, w), (box[1] + box[3]) / 2.0 / max(1.0, h)),
        })
    labels = sorted(labels, key=lambda x: (x['box'][1], x['box'][0]))
    return labels


def _box_center(box: Tuple[int, int, int, int]) -> Tuple[float, float]:
    x0, y0, x1, y1 = box
    return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)


def _box_distance(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax, ay = _box_center(a)
    bx, by = _box_center(b)
    return float(((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5)


def _associate_labels_to_panels(panels: Sequence[Tuple[int, int, int, int]], labels: Sequence[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    mapping: Dict[int, Dict[str, Any]] = {}
    if not panels or not labels:
        return mapping
    used = set()
    for panel_idx, panel in enumerate(panels):
        best = None
        best_score = float('inf')
        px0, py0, px1, py1 = panel
        for label_idx, label in enumerate(labels):
            if label_idx in used:
                continue
            box = tuple(label['box'])
            lx0, ly0, lx1, ly1 = box
            penalty = 0.0
            if ly1 > py1:
                penalty += (ly1 - py1) * 1.5
            if lx1 < px0:
                penalty += (px0 - lx1) * 0.5
            if lx0 > px1:
                penalty += (lx0 - px1) * 0.5
            score = _box_distance(panel, box) + penalty
            if score < best_score:
                best = label_idx
                best_score = score
        if best is not None:
            mapping[panel_idx] = labels[best]
            used.add(best)
    return mapping


def _sort_labels_reading_order(labels: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(labels, key=lambda item: (item['box'][1], item['box'][0]))


def _merge_boxes(boxes: Sequence[Tuple[int, int, int, int]], image_shape: Tuple[int, int, int], pad_scale: float = 0.05) -> Optional[Tuple[int, int, int, int]]:
    if not boxes:
        return None
    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    x1 = max(b[2] for b in boxes)
    y1 = max(b[3] for b in boxes)
    h, w = image_shape[:2]
    pad_x = int((x1 - x0) * pad_scale)
    pad_y = int((y1 - y0) * pad_scale)
    return (max(0, x0 - pad_x), max(0, y0 - pad_y), min(w, x1 + pad_x), min(h, y1 + pad_y))


def _foreground_mask_grabcut(rgb: np.ndarray, inset: float = 0.04, iterations: int = 5) -> np.ndarray:
    h, w = rgb.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    mx = max(1, int(w * inset))
    my = max(1, int(h * inset))
    rect = (mx, my, max(1, w - 2 * mx), max(1, h - 2 * my))
    try:
        cv2.grabCut(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), mask, rect, bgd, fgd, iterations, cv2.GC_INIT_WITH_RECT)
        fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)
        return fg
    except Exception:
        return np.zeros((h, w), np.uint8)


def _saturation_mask(rgb: np.ndarray, sat_threshold: int = 36) -> np.ndarray:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    s = hsv[..., 1]
    mask = (s >= sat_threshold).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    return mask


def _texture_saliency_mask(rgb: np.ndarray, std_threshold: float = 20.0) -> np.ndarray:
    gray = _rgb_to_gray(rgb).astype(np.float32)
    mean = cv2.GaussianBlur(gray, (0, 0), 4.0)
    mean_sq = cv2.GaussianBlur(gray * gray, (0, 0), 4.0)
    std = np.sqrt(np.maximum(mean_sq - mean * mean, 0))
    mask = (std >= std_threshold).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    return mask


def _border_contact_ratio(mask: np.ndarray, band: int = 10) -> float:
    h, w = mask.shape[:2]
    band = max(1, min(band, h // 4 if h > 4 else 1, w // 4 if w > 4 else 1))
    border = np.zeros_like(mask, dtype=np.uint8)
    border[:band, :] = 1
    border[-band:, :] = 1
    border[:, :band] = 1
    border[:, -band:] = 1
    denom = float((mask > 0).sum()) + EPS
    return float(np.logical_and(mask > 0, border > 0).sum()) / denom


def _candidate_object_boxes(collage_rgb: np.ndarray, labels: Optional[Sequence[Dict[str, Any]]] = None) -> List[Tuple[int, int, int, int]]:
    work = collage_rgb.copy()
    label_boxes = [tuple(item['box']) for item in labels] if labels else _detect_label_boxes(work)
    if label_boxes:
        bg = _estimate_border_bg_color(work).astype(np.uint8)
        for x0, y0, x1, y1 in label_boxes:
            work[y0:y1, x0:x1] = bg

    masks = []
    gc_mask = _foreground_mask_grabcut(work, inset=0.05, iterations=4)
    if gc_mask.sum() > 0:
        masks.append(gc_mask)
    masks.append(_saturation_mask(work, sat_threshold=36))
    masks.append(_texture_saliency_mask(work, std_threshold=18.0))
    masks.append(_background_distance_mask(work, threshold=24.0))
    masks.append(_white_bg_foreground_mask(work))

    h, w = work.shape[:2]
    min_area = max(800, (h * w) // 400)
    boxes: List[Tuple[int, int, int, int]] = []
    for mask in masks:
        comp_boxes = _connected_components_boxes(mask, min_area=min_area)
        for box in comp_boxes:
            bx0, by0, bx1, by1 = box
            bw = bx1 - bx0
            bh = by1 - by0
            area = bw * bh
            if bw < max(24, w * 0.05) or bh < max(24, h * 0.05):
                continue
            if area > h * w * 0.98:
                continue
            boxes.append(box)

    # dedupe near-identical boxes by IoU-like overlap
    uniq: List[Tuple[int, int, int, int]] = []
    for box in sorted(boxes, key=lambda b: (b[0], b[1], -(b[2] - b[0]) * (b[3] - b[1]))):
        keep = True
        for existing in uniq:
            ix0 = max(box[0], existing[0])
            iy0 = max(box[1], existing[1])
            ix1 = min(box[2], existing[2])
            iy1 = min(box[3], existing[3])
            if ix1 <= ix0 or iy1 <= iy0:
                continue
            inter = (ix1 - ix0) * (iy1 - iy0)
            area_box = (box[2] - box[0]) * (box[3] - box[1])
            area_existing = (existing[2] - existing[0]) * (existing[3] - existing[1])
            denom = float(area_box + area_existing - inter) + EPS
            if inter / denom > 0.7:
                keep = False
                break
        if keep:
            uniq.append(box)
    return uniq


def _crop_to_mask(rgb: np.ndarray, mask: np.ndarray, pad_scale: float = 0.08) -> Tuple[np.ndarray, np.ndarray, Optional[Tuple[int, int, int, int]]]:
    mask = (mask > 0).astype(np.uint8)
    bbox = _bbox_from_mask(mask)
    if bbox is None:
        h, w = rgb.shape[:2]
        return rgb.copy(), np.ones((h, w), np.uint8), (0, 0, w, h)
    bbox = _expand_bbox(bbox, 1.0 + pad_scale, rgb.shape)
    crop = _crop(rgb, bbox)
    crop_mask = _crop(mask, bbox)
    return crop, crop_mask, bbox


def _masked_pixels(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    mask = (mask > 0)
    if mask.sum() < 9:
        return rgb.reshape(-1, 3)
    return rgb[mask]


def _masked_hist_similarity_rgb(a: np.ndarray, a_mask: np.ndarray, b: np.ndarray, b_mask: np.ndarray, bins: int = 16) -> float:
    sims = []
    for ch in range(3):
        hist_a = cv2.calcHist([a], [ch], (a_mask > 0).astype(np.uint8), [bins], [0, 256]).reshape(-1)
        hist_b = cv2.calcHist([b], [ch], (b_mask > 0).astype(np.uint8), [bins], [0, 256]).reshape(-1)
        hist_a /= float(hist_a.sum()) + EPS
        hist_b /= float(hist_b.sum()) + EPS
        sims.append(_clamp01(cv2.compareHist(hist_a.astype(np.float32), hist_b.astype(np.float32), cv2.HISTCMP_CORREL)))
    return float(np.mean(sims))


def _masked_lab_color_similarity(a: np.ndarray, a_mask: np.ndarray, b: np.ndarray, b_mask: np.ndarray) -> float:
    a_lab = _rgb2lab_image(a)
    b_lab = _rgb2lab_image(b)
    pa = _masked_pixels(a_lab, a_mask)
    pb = _masked_pixels(b_lab, b_mask)
    mean_a = pa.reshape(-1, 3).mean(axis=0)
    mean_b = pb.reshape(-1, 3).mean(axis=0)
    dist = np.linalg.norm(mean_a - mean_b)
    return _clamp01(1.0 - dist / 55.0)


def _masked_texture_similarity(a: np.ndarray, a_mask: np.ndarray, b: np.ndarray, b_mask: np.ndarray) -> float:
    a = _resize_rgb(a, (128, 128))
    b = _resize_rgb(b, (128, 128))
    a_mask = cv2.resize((a_mask > 0).astype(np.uint8), (128, 128), interpolation=cv2.INTER_NEAREST)
    b_mask = cv2.resize((b_mask > 0).astype(np.uint8), (128, 128), interpolation=cv2.INTER_NEAREST)
    ga = _rgb_to_gray(a)
    gb = _rgb_to_gray(b)
    ga = np.where(a_mask > 0, ga, int(np.median(ga[a_mask > 0])) if (a_mask > 0).sum() > 0 else ga)
    gb = np.where(b_mask > 0, gb, int(np.median(gb[b_mask > 0])) if (b_mask > 0).sum() > 0 else gb)
    lbp_a = _lbp_hist(ga)
    lbp_b = _lbp_hist(gb)
    ssim_val = _ssim_score(ga, gb)
    return _clamp01(0.60 * _cosine_similarity(lbp_a, lbp_b) + 0.40 * ((ssim_val + 1.0) / 2.0))


def _masked_edge_similarity(a: np.ndarray, a_mask: np.ndarray, b: np.ndarray, b_mask: np.ndarray) -> float:
    a = _resize_rgb(a, (160, 160))
    b = _resize_rgb(b, (160, 160))
    a_mask = cv2.resize((a_mask > 0).astype(np.uint8), (160, 160), interpolation=cv2.INTER_NEAREST)
    b_mask = cv2.resize((b_mask > 0).astype(np.uint8), (160, 160), interpolation=cv2.INTER_NEAREST)
    ga = _rgb_to_gray(a)
    gb = _rgb_to_gray(b)
    ea = cv2.Canny(ga, 60, 160)
    eb = cv2.Canny(gb, 60, 160)
    ea = np.where(a_mask > 0, ea, 0)
    eb = np.where(b_mask > 0, eb, 0)
    inter = float(np.logical_and(ea > 0, eb > 0).sum())
    union = float(np.logical_or(ea > 0, eb > 0).sum()) + EPS
    iou = inter / union
    density_sim = 1.0 - abs(float((ea > 0).mean()) - float((eb > 0).mean()))
    return _clamp01(0.70 * iou + 0.30 * density_sim)


def _rotate_image_and_mask(rgb: np.ndarray, mask: np.ndarray, angle_deg: float) -> Tuple[np.ndarray, np.ndarray]:
    h, w = rgb.shape[:2]
    center = (w / 2.0, h / 2.0)
    mat = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    rotated_rgb = cv2.warpAffine(rgb, mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    rotated_mask = cv2.warpAffine((mask > 0).astype(np.uint8), mat, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return rotated_rgb, rotated_mask


def _best_masked_edge_similarity(ref_rgb: np.ndarray, ref_mask: np.ndarray, target_rgb: np.ndarray, target_mask: np.ndarray, garment_type: str = 'unknown') -> float:
    garment_type = (garment_type or 'unknown').lower().strip()
    angles = [0.0]
    if garment_type in {'hat', 'shoes', 'purse', 'bag'}:
        angles = [-20.0, -10.0, 0.0, 10.0, 20.0]
    scores = []
    for angle in angles:
        if abs(angle) < 1e-6:
            cand_rgb, cand_mask = target_rgb, target_mask
        else:
            cand_rgb, cand_mask = _rotate_image_and_mask(target_rgb, target_mask, angle)
        scores.append(_masked_edge_similarity(ref_rgb, ref_mask, cand_rgb, cand_mask))
    return max(scores) if scores else 0.0


def _target_mask_from_region(region_rgb: np.ndarray, garment_type: str, ref_crop: Optional[np.ndarray] = None) -> np.ndarray:
    masks = []
    gc = _foreground_mask_grabcut(region_rgb, inset=0.05, iterations=4)
    if gc.sum() > 0:
        masks.append(gc)
    masks.append(_saturation_mask(region_rgb, sat_threshold=28))
    masks.append(_texture_saliency_mask(region_rgb, std_threshold=16.0))
    masks.append(_background_distance_mask(region_rgb, threshold=24.0))
    masks.append(_white_bg_foreground_mask(region_rgb))

    if ref_crop is not None:
        ref_lab = _rgb2lab_image(ref_crop)
        ref_mean = ref_lab.reshape(-1, 3).mean(axis=0)
        region_lab = _rgb2lab_image(region_rgb)
        dist = np.linalg.norm(region_lab - ref_mean[None, None, :], axis=2)
        color_mask = (dist < np.percentile(dist, 55)).astype(np.uint8)
        masks.append(color_mask)

    combined = np.zeros(region_rgb.shape[:2], np.uint8)
    for mask in masks:
        combined = np.maximum(combined, (mask > 0).astype(np.uint8))

    h, w = combined.shape[:2]
    comps = _connected_components_boxes(combined, min_area=max(120, (h * w) // 150))
    if not comps:
        return _largest_connected_component(combined) if combined.sum() > 0 else combined

    # score candidates using expected garment geometry
    candidates = []
    for box in comps:
        x0, y0, x1, y1 = box
        bw = x1 - x0
        bh = y1 - y0
        area = max(1, bw * bh)
        sub = combined[y0:y1, x0:x1]
        fill = float(sub.mean())
        aspect = bw / max(1.0, bh)
        cx = (x0 + x1) / 2.0 / max(1.0, w)
        cy = (y0 + y1) / 2.0 / max(1.0, h)
        score = 0.0
        if garment_type == 'hat':
            score += 2.2 * (1.0 - min(1.0, abs(cy - 0.26) / 0.35))
            score += 1.3 * min(1.0, aspect / 1.8)
            score += 0.7 * fill
        elif garment_type == 'shoes':
            score += 2.0 * (1.0 - min(1.0, abs(cy - 0.92) / 0.25))
            score += 0.8 * min(1.0, aspect / 2.0)
            score += 0.6 * fill
        else:
            score += 1.0 * fill + 0.3 * min(1.0, area / float(h * w * 0.25))
        candidates.append((score, box))
    best_box = max(candidates, key=lambda item: item[0])[1]
    out = np.zeros_like(combined)
    x0, y0, x1, y1 = best_box
    out[y0:y1, x0:x1] = combined[y0:y1, x0:x1]
    out = _largest_connected_component(out)
    return out



def _extract_reference_object_crop(panel_rgb: np.ndarray, garment_type: str = 'unknown') -> Tuple[np.ndarray, np.ndarray, Optional[Tuple[int, int, int, int]]]:
    work = panel_rgb.copy()
    local_labels = _detect_label_boxes(work)
    if local_labels:
        bg = np.median(work.reshape(-1, 3), axis=0).astype(np.uint8)
        for x0, y0, x1, y1 in local_labels:
            work[y0:y1, x0:x1] = bg

    masks = []
    gc = _foreground_mask_grabcut(work, inset=0.05, iterations=5)
    if gc.sum() > 0:
        masks.append(gc)
    masks.append(_saturation_mask(work, sat_threshold=34))
    masks.append(_texture_saliency_mask(work, std_threshold=18.0))
    masks.append(_background_distance_mask(work, threshold=22.0))
    masks.append(_white_bg_foreground_mask(work))

    garment_type = (garment_type or 'unknown').lower().strip()
    best_mask = None
    best_score = -1.0
    h, w = work.shape[:2]
    hsv = cv2.cvtColor(work, cv2.COLOR_RGB2HSV)
    sat = hsv[..., 1].astype(np.float32) / 255.0
    for mask in masks:
        if mask.sum() == 0:
            continue
        cc_boxes = _connected_components_boxes(mask, min_area=max(200, (h * w) // 200))
        if not cc_boxes:
            cc_boxes = [_bbox_from_mask(mask)] if _bbox_from_mask(mask) is not None else []
        for bbox in cc_boxes:
            if bbox is None:
                continue
            x0, y0, x1, y1 = bbox
            bw = x1 - x0
            bh = y1 - y0
            area = max(1, bw * bh)
            sub_mask = (mask[y0:y1, x0:x1] > 0).astype(np.uint8)
            fill = float(sub_mask.mean())
            aspect = bw / max(1.0, bh)
            center_penalty = abs(((x0 + x1) / 2.0) / max(1.0, w) - 0.5) + abs(((y0 + y1) / 2.0) / max(1.0, h) - 0.5)
            thin_penalty = 1.0 if min(bw, bh) < max(14, min(h, w) * 0.06) else 0.0
            border_contact = _border_contact_ratio(sub_mask, band=max(8, min(h, w) // 40))
            inside_sat = float(sat[y0:y1, x0:x1][sub_mask > 0].mean()) if (sub_mask > 0).sum() > 0 else 0.0
            outside_mask = np.ones_like(mask[y0:y1, x0:x1], dtype=np.uint8)
            outside_mask[sub_mask > 0] = 0
            outside_sat = float(sat[y0:y1, x0:x1][outside_mask > 0].mean()) if (outside_mask > 0).sum() > 0 else 0.0
            sat_contrast = max(0.0, inside_sat - outside_sat)
            oversize_penalty = 1.0 if (bw > w * 0.96 or bh > h * 0.96) else 0.0
            score = (
                1.0 * min(1.0, area / float(h * w * 0.45))
                + 0.9 * fill
                + 0.7 * min(1.0, aspect / 2.5)
                + 1.0 * min(1.0, sat_contrast / 0.25)
                - 0.6 * center_penalty
                - 0.9 * thin_penalty
                - 1.4 * border_contact
                - 1.2 * oversize_penalty
            )
            if garment_type == 'hat':
                score += 0.5 * min(1.0, aspect / 1.6)
                score += 0.4 * (1.0 - min(1.0, abs(((y0 + y1) / 2.0) / max(1.0, h) - 0.55) / 0.55))
            candidate_mask = np.zeros_like(mask)
            candidate_mask[y0:y1, x0:x1] = sub_mask
            if score > best_score:
                best_score = score
                best_mask = candidate_mask

    if best_mask is None or best_mask.sum() == 0:
        mask = _white_bg_foreground_mask(work)
    else:
        mask = _largest_connected_component(best_mask)
    return _crop_to_mask(work, mask, pad_scale=0.08)


def _split_collage_panels(collage_rgb: np.ndarray, labels: Optional[Sequence[Dict[str, Any]]] = None) -> List[Tuple[int, int, int, int]]:
    h, w = collage_rgb.shape[:2]
    labels = _sort_labels_reading_order(labels if labels is not None else _parse_collage_labels(collage_rgb))
    object_boxes = _candidate_object_boxes(collage_rgb, labels)

    if labels:
        # Single labeled garment image: whole image is the panel.
        if len(labels) == 1:
            if object_boxes:
                merged = _merge_boxes(object_boxes, collage_rgb.shape, pad_scale=0.08)
                return [merged] if merged is not None else [(0, 0, w, h)]
            return [(0, 0, w, h)]

        # Multiple labels: prefer one object box per label, associated by proximity.
        if object_boxes:
            available = list(object_boxes)
            panels: List[Tuple[int, int, int, int]] = []
            for label in labels:
                lbox = tuple(label['box'])
                best_idx = None
                best_score = float('inf')
                for idx, box in enumerate(available):
                    score = _box_distance(lbox, box)
                    if score < best_score:
                        best_score = score
                        best_idx = idx
                if best_idx is not None:
                    box = available.pop(best_idx)
                    panels.append(_expand_bbox(box, 1.06, collage_rgb.shape))
            if len(panels) == len(labels):
                return panels

        # Fallback: derive panel slices from label positions.
        ordered = labels
        centers = [int((item['box'][0] + item['box'][2]) / 2.0) for item in ordered]
        boundaries = [0]
        for a, b in zip(centers[:-1], centers[1:]):
            boundaries.append(int((a + b) / 2.0))
        boundaries.append(w)
        panels = []
        for idx in range(len(ordered)):
            x0, x1 = boundaries[idx], boundaries[idx + 1]
            panels.append((max(0, x0), 0, min(w, x1), h))
        return panels

    # No labels: fallback to object-driven panels.
    if object_boxes:
        return sorted(object_boxes, key=lambda b: (b[1], b[0]))

    gray = _rgb_to_gray(collage_rgb)
    active_cols = np.where(gray.mean(axis=0) < 250)[0]
    active_rows = np.where(gray.mean(axis=1) < 250)[0]
    if len(active_cols) == 0 or len(active_rows) == 0:
        return [(0, 0, w, h)]
    x0, x1 = int(active_cols.min()), int(active_cols.max()) + 1
    y0, y1 = int(active_rows.min()), int(active_rows.max()) + 1
    return [(x0, y0, x1, y1)]


def _infer_garment_type_heuristic(rgb: np.ndarray) -> str:
    h, w = rgb.shape[:2]
    aspect = w / max(1.0, h)
    mask = _white_bg_foreground_mask(rgb)
    bbox = _bbox_from_mask(mask)
    fill = float(mask.mean())
    if bbox is None:
        return "unknown"
    bx0, by0, bx1, by1 = bbox
    bw = bx1 - bx0
    bh = by1 - by0
    aspect_fg = bw / max(1.0, bh)
    if bh < h * 0.28 and aspect_fg > 1.2:
        return "hat"
    if bh < h * 0.35 and aspect_fg > 1.6:
        return "shoes"
    if aspect < 0.7 and fill > 0.35:
        return "dress"
    if aspect < 0.8 and fill > 0.2:
        return "pants"
    if aspect_fg > 0.9 and bh < h * 0.65:
        return "purse"
    return "shirt"


def _expected_body_region(person_bbox: Tuple[int, int, int, int], garment_type: str, image_shape: Tuple[int, int, int]) -> Tuple[int, int, int, int]:
    x0, y0, x1, y1 = person_bbox
    w = x1 - x0
    h = y1 - y0

    def box(rx0: float, ry0: float, rx1: float, ry1: float) -> Tuple[int, int, int, int]:
        return (
            int(x0 + w * rx0),
            int(y0 + h * ry0),
            int(x0 + w * rx1),
            int(y0 + h * ry1),
        )

    garment_type = (garment_type or "unknown").lower()
    if garment_type == "hat":
        return box(0.08, -0.04, 0.92, 0.27)
    if garment_type in {"shirt", "jacket", "top", "blouse"}:
        return box(0.12, 0.14, 0.88, 0.58)
    if garment_type in {"pants", "skirt"}:
        return box(0.16, 0.46, 0.84, 0.96)
    if garment_type == "dress":
        return box(0.12, 0.12, 0.88, 0.96)
    if garment_type == "shoes":
        return box(0.10, 0.84, 0.90, 1.02)
    if garment_type in {"purse", "bag"}:
        return box(0.54, 0.20, 0.98, 0.76)
    return box(0.12, 0.14, 0.88, 0.88)


def _presence_threshold_for_type(garment_type: str) -> float:
    garment_type = (garment_type or 'unknown').lower()
    if garment_type == 'hat':
        return 0.24
    if garment_type == 'shoes':
        return 0.28
    if garment_type in {'purse', 'bag'}:
        return 0.26
    return 0.32


def _mask_aspect_similarity(ref_mask: np.ndarray, target_mask: np.ndarray) -> float:
    rb = _bbox_from_mask(ref_mask)
    tb = _bbox_from_mask(target_mask)
    if rb is None or tb is None:
        return 0.0
    ra = (rb[2] - rb[0]) / max(1.0, (rb[3] - rb[1]))
    ta = (tb[2] - tb[0]) / max(1.0, (tb[3] - tb[1]))
    return _clamp01(1.0 - abs(math.log(max(ra, EPS)) - math.log(max(ta, EPS))) / 0.8)


def _mask_fill_similarity(ref_mask: np.ndarray, target_mask: np.ndarray) -> float:
    rf = float(ref_mask.mean())
    tf = float(target_mask.mean())
    return _clamp01(1.0 - abs(rf - tf) / 0.8)


def _body_width_profile(mask: np.ndarray, n: int = 12) -> np.ndarray:
    bbox = _bbox_from_mask(mask)
    if bbox is None:
        return np.zeros((n,), dtype=np.float32)
    x0, y0, x1, y1 = bbox
    crop = mask[y0:y1, x0:x1]
    if crop.shape[0] < n:
        crop = cv2.resize(crop.astype(np.uint8), (crop.shape[1], n), interpolation=cv2.INTER_NEAREST)
    positions = np.linspace(0, crop.shape[0] - 1, n).astype(int)
    widths = []
    for y in positions:
        row = crop[y]
        widths.append(float(row.sum()) / max(1.0, crop.shape[1]))
    return np.asarray(widths, dtype=np.float32)


def _body_ratio_similarity(src_mask: np.ndarray, dst_mask: np.ndarray) -> float:
    src_prof = _body_width_profile(src_mask)
    dst_prof = _body_width_profile(dst_mask)
    return _cosine_similarity(src_prof, dst_prof)


def _silhouette_similarity(src_mask: np.ndarray, dst_mask: np.ndarray) -> float:
    src = cv2.resize((src_mask > 0).astype(np.uint8), (128, 256), interpolation=cv2.INTER_NEAREST)
    dst = cv2.resize((dst_mask > 0).astype(np.uint8), (128, 256), interpolation=cv2.INTER_NEAREST)
    inter = float(np.logical_and(src > 0, dst > 0).sum())
    union = float(np.logical_or(src > 0, dst > 0).sum()) + EPS
    return _clamp01(inter / union)


def _lighting_class(rgb: np.ndarray) -> Tuple[str, Dict[str, float]]:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    v = hsv[..., 2].astype(np.float32) / 255.0
    s = hsv[..., 1].astype(np.float32) / 255.0
    mean_v = float(v.mean())
    std_v = float(v.std())
    mean_s = float(s.mean())
    mean_rgb = rgb.reshape(-1, 3).mean(axis=0)
    warmth = float((mean_rgb[0] - mean_rgb[2]) / 255.0)
    gray = _rgb_to_gray(rgb)
    lap = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    features = {
        "brightness": mean_v,
        "contrast": std_v,
        "saturation": mean_s,
        "warmth": warmth,
        "clarity": min(1.0, lap / 350.0),
    }

    if mean_v > 0.76 and std_v < 0.17 and mean_s < 0.22:
        return "photoshoot", features
    if mean_v > 0.72 and warmth > 0.08:
        return "golden_hour", features
    if mean_v > 0.78 and std_v > 0.22:
        return "hot_afternoon_sun", features
    if mean_v < 0.68 and std_v < 0.12 and mean_s < 0.25:
        return "foggy_phone_pic", features
    return "natural_light", features


def _lighting_class_similarity(predicted: str, expected: str) -> float:
    if not expected:
        return 0.7
    predicted = predicted.lower().strip()
    expected = expected.lower().strip()
    if predicted == expected:
        return 1.0
    near = {
        ("photoshoot", "natural_light"),
        ("natural_light", "photoshoot"),
        ("hot_afternoon_sun", "golden_hour"),
        ("golden_hour", "hot_afternoon_sun"),
        ("natural_light", "golden_hour"),
        ("golden_hour", "natural_light"),
    }
    return 0.65 if (predicted, expected) in near else 0.2


def _lighting_consistency(rgb: np.ndarray, person_mask: Optional[np.ndarray] = None) -> float:
    gray = _rgb_to_gray(rgb).astype(np.float32) / 255.0
    brightness = float(gray.mean())
    contrast = float(gray.std())
    smoothness = float(cv2.GaussianBlur(gray, (0, 0), 1.0).std())
    score = 0.45 * (1.0 - min(1.0, abs(brightness - 0.6) / 0.6))
    score += 0.35 * (1.0 - min(1.0, abs(contrast - 0.22) / 0.22))
    score += 0.20 * min(1.0, smoothness / 0.22)
    if person_mask is not None and person_mask.mean() > 0:
        bg_mask = (1 - (person_mask > 0).astype(np.uint8))
        if bg_mask.sum() > 0:
            person_mean = float(gray[person_mask > 0].mean())
            bg_mean = float(gray[bg_mask > 0].mean())
            score *= _clamp01(1.0 - abs(person_mean - bg_mean) / 0.65)
    return _clamp01(score)


def _background_scores(rgb: np.ndarray, person_mask: np.ndarray) -> Dict[str, float]:
    bg_mask = (person_mask == 0)
    if bg_mask.sum() < 64:
        return {"scene_plausibility": 0.4, "subject_background_integration": 0.4, "composition_balance": 0.5}
    bg = rgb.copy()
    bg_pixels = bg[bg_mask]
    gray = _rgb_to_gray(rgb)
    bg_gray = gray[bg_mask]

    # plausibility: avoid excessive noise / clipping / empty failure regions
    noise = float(np.std(bg_gray / 255.0))
    clipped = float(np.mean((bg_gray < 5) | (bg_gray > 250)))
    scene_plausibility = _clamp01(0.55 * (1.0 - min(1.0, abs(noise - 0.20) / 0.35)) + 0.45 * (1.0 - clipped))

    # integration: edge halo around foreground boundary
    boundary = cv2.Canny((person_mask * 255).astype(np.uint8), 50, 150) > 0
    dilated = cv2.dilate(boundary.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1) > 0
    boundary_vals = gray[dilated]
    full_std = float(gray.std()) + EPS
    integration = _clamp01(1.0 - abs(float(boundary_vals.std()) - full_std) / max(full_std, 1.0))

    # composition: subject centered but not overly cropped
    bbox = _bbox_from_mask(person_mask)
    if bbox is None:
        composition_balance = 0.5
    else:
        x0, y0, x1, y1 = bbox
        h, w = gray.shape[:2]
        cx = ((x0 + x1) / 2.0) / max(1.0, w)
        cy = ((y0 + y1) / 2.0) / max(1.0, h)
        crop_margin = min(x0, y0, w - x1, h - y1) / max(1.0, min(h, w))
        center_score = 1.0 - min(1.0, abs(cx - 0.5) / 0.5) * 0.5 - min(1.0, abs(cy - 0.55) / 0.55) * 0.5
        composition_balance = _clamp01(0.7 * center_score + 0.3 * min(1.0, crop_margin / 0.1))

    return {
        "scene_plausibility": scene_plausibility,
        "subject_background_integration": integration,
        "composition_balance": composition_balance,
    }


def _default_rubric() -> Dict[str, Any]:
    return {
        "version": "1.0",
        "options": {
            "renormalize_weights": True,
            "allow_best_guess_labels": False,
            "inventory_mode": "rubric_only",
            "use_collage_labels_when_present": True,
            "speed_profile": "balanced",
        },
        "top_level_sections": {
            "face_identity": {
                "enabled": True,
                "weight": 0.22,
                "subsections": {
                    "identity_embedding": {"enabled": True, "weight": 0.80},
                    "landmark_geometry": {"enabled": True, "weight": 0.10},
                    "face_quality_visibility": {"enabled": True, "weight": 0.10},
                },
            },
            "garments_total": {
                "enabled": True,
                "weight": 0.46,
                "hard_fail_on_missing_required": True,
                "default_subsections": {
                    "presence": {"enabled": True, "weight": 0.35},
                    "shape_structure": {"enabled": True, "weight": 0.30},
                    "texture_color_text": {"enabled": True, "weight": 0.25},
                    "placement": {"enabled": True, "weight": 0.10},
                },
                "inventory": [],
            },
            "body_shape": {
                "enabled": True,
                "weight": 0.14,
                "subsections": {
                    "ratio_similarity": {"enabled": True, "weight": 0.65},
                    "silhouette_similarity": {"enabled": True, "weight": 0.35},
                },
            },
            "lighting": {
                "enabled": True,
                "weight": 0.10,
                "desired_class": "",
                "subsections": {
                    "lighting_class_match": {"enabled": True, "weight": 0.45},
                    "photometric_consistency": {"enabled": True, "weight": 0.55},
                },
            },
            "background": {
                "enabled": True,
                "weight": 0.08,
                "subsections": {
                    "scene_plausibility": {"enabled": True, "weight": 0.55},
                    "subject_background_integration": {"enabled": True, "weight": 0.30},
                    "composition_balance": {"enabled": True, "weight": 0.15},
                },
            },
        },
    }


def _deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(a)
    for key, value in b.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _validate_rubric(rubric: Dict[str, Any], collage_rgb: Optional[np.ndarray] = None) -> Dict[str, Any]:
    base = _default_rubric()
    merged = _deep_merge(base, rubric)
    sections = merged.get("top_level_sections", {})
    if "garments_total" not in sections:
        sections["garments_total"] = base["top_level_sections"]["garments_total"]

    inventory = sections["garments_total"].get("inventory", [])
    options = merged.get("options", {})
    inventory_mode = str(options.get("inventory_mode", "rubric_only")).lower()
    allow_best_guess = bool(options.get("allow_best_guess_labels", False))
    use_collage_labels = bool(options.get("use_collage_labels_when_present", True))

    labels: List[Dict[str, Any]] = []
    panels: List[Tuple[int, int, int, int]] = []
    panel_to_label: Dict[int, Dict[str, Any]] = {}

    if collage_rgb is not None:
        labels = _parse_collage_labels(collage_rgb)
        panels = _split_collage_panels(collage_rgb, labels=labels)
        panel_to_label = _associate_labels_to_panels(panels, labels)

        for idx, item in enumerate(inventory):
            if item.get("panel_index") is None:
                item["panel_index"] = idx if idx < len(panels) else max(0, len(panels) - 1)
            panel_idx = int(item.get("panel_index", 0)) if panels else 0
            panel_idx = max(0, min(panel_idx, max(0, len(panels) - 1)))
            label_info = panel_to_label.get(panel_idx)
            if use_collage_labels and label_info is not None:
                item["garment_type"] = label_info.get("primary_label", item.get("garment_type", "unknown"))
                item["label_source"] = "collage_ocr"
            elif not item.get("garment_type"):
                if allow_best_guess and panels:
                    item["garment_type"] = _infer_garment_type_heuristic(_crop(collage_rgb, panels[panel_idx]))
                    item["label_source"] = "best_guess"
                else:
                    item["garment_type"] = "unknown"
                    item["label_source"] = "none"

        should_autogenerate = False
        if not inventory and panels:
            if labels and use_collage_labels:
                should_autogenerate = True
            elif inventory_mode in {"best_guess_if_empty", "auto_from_collage", "collage_labels_first", "labels_only"}:
                should_autogenerate = True

        if should_autogenerate:
            generated = []
            for idx, panel in enumerate(panels):
                label_info = panel_to_label.get(idx)
                if label_info is not None and use_collage_labels:
                    garment_type = label_info.get("primary_label", "unknown")
                    label_source = "collage_ocr"
                else:
                    crop = _crop(collage_rgb, panel)
                    garment_type = _infer_garment_type_heuristic(crop) if allow_best_guess else "unknown"
                    label_source = "best_guess" if allow_best_guess else "none"
                generated.append({
                    "garment_id": f"{garment_type or 'garment'}_{idx+1}",
                    "garment_type": garment_type or "unknown",
                    "required": True,
                    "weight_within_garments": 1.0 / max(1, len(panels)),
                    "panel_index": idx,
                    "auto_generated": True,
                    "label_source": label_source,
                })
            inventory = generated
            sections["garments_total"]["inventory"] = inventory

    merged.setdefault("meta", {})["parsed_collage_labels"] = labels
    merged.setdefault("meta", {})["detected_panels"] = panels

    if inventory:
        enabled_weights = []
        for item in inventory:
            if bool(item.get("enabled", True)):
                enabled_weights.append(_safe_float(item.get("weight_within_garments", 1.0), 1.0))
        total = sum(enabled_weights) if enabled_weights else float(len(inventory))
        for item in inventory:
            if bool(item.get("enabled", True)):
                weight = _safe_float(item.get("weight_within_garments", 1.0), 1.0)
                item["normalized_weight_within_garments"] = weight / max(total, EPS)
            else:
                item["normalized_weight_within_garments"] = 0.0
    return merged


def _report_shell() -> Dict[str, Any]:
    return {
        "final_score": 0.0,
        "hard_fail": False,
        "hard_fail_reasons": [],
        "top_level_scores": {},
        "top_level_confidence": {},
        "top_level_weights_used": {},
        "garments": [],
        "meta": {},
        "operator_recommendation": "reject",
    }


class VTONRubricLoadNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "rubric_path": ("STRING", {"multiline": False, "default": ""}),
                "rubric_json_override": ("STRING", {"multiline": True, "default": ""}),
                "collage_image": ("IMAGE", {}),
            }
        }

    RETURN_TYPES = ("VTON_RUBRIC", "STRING")
    RETURN_NAMES = ("rubric", "rubric_json")
    FUNCTION = "load_rubric"
    CATEGORY = PACKAGE_CATEGORY

    def load_rubric(self, rubric_path: str, rubric_json_override: str, collage_image: torch.Tensor):
        rubric_data: Dict[str, Any] = {}
        if rubric_json_override and rubric_json_override.strip():
            rubric_data = json.loads(rubric_json_override)
        elif rubric_path and rubric_path.strip():
            with open(rubric_path, "r", encoding="utf-8") as f:
                rubric_data = json.load(f)
        collage_rgb = _tensor_to_np(collage_image)
        rubric = _validate_rubric(rubric_data, collage_rgb=collage_rgb)
        return (rubric, _json_dumps(rubric))


class VTONFaceIdentityScoreNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_image": ("IMAGE", {}),
                "tryon_image": ("IMAGE", {}),
                "rubric": ("VTON_RUBRIC", {}),
            }
        }

    RETURN_TYPES = ("VTON_SECTION_SCORE", "FLOAT", "STRING")
    RETURN_NAMES = ("face_section", "face_score", "face_json")
    FUNCTION = "score_face"
    CATEGORY = PACKAGE_CATEGORY

    def score_face(self, source_image: torch.Tensor, tryon_image: torch.Tensor, rubric: Dict[str, Any]):
        src = _tensor_to_np(source_image)
        dst = _tensor_to_np(tryon_image)
        section = rubric["top_level_sections"]["face_identity"]
        if not section.get("enabled", True):
            payload = {"enabled": False, "score": 0.0, "confidence": 0.0, "subscores": {}}
            return (payload, 0.0, _json_dumps(payload))

        subweights = _normalize_subweights(section.get("subsections", {}))
        feat_src = _face_feature_vector(src)
        feat_dst = _face_feature_vector(dst)
        bbox_src, conf_src = _extract_face_bbox(src)
        bbox_dst, conf_dst = _extract_face_bbox(dst)
        identity_embedding = _cosine_similarity(feat_src, feat_dst)
        landmark_geometry = _face_geometry_score(src, dst)
        crop_src = _crop(src, _expand_bbox(bbox_src, 1.2, src.shape))
        crop_dst = _crop(dst, _expand_bbox(bbox_dst, 1.2, dst.shape))
        face_quality_visibility = float(np.mean([
            conf_src,
            conf_dst,
            _blur_score(crop_src),
            _blur_score(crop_dst),
        ]))
        face_confidence = _clamp01(face_quality_visibility)

        score = (
            subweights.get("identity_embedding", 0.0) * identity_embedding
            + subweights.get("landmark_geometry", 0.0) * landmark_geometry
            + subweights.get("face_quality_visibility", 0.0) * face_quality_visibility
        ) * face_confidence

        payload = {
            "enabled": True,
            "score": _clamp01(score),
            "confidence": face_confidence,
            "subscores": {
                "identity_embedding": identity_embedding,
                "landmark_geometry": landmark_geometry,
                "face_quality_visibility": face_quality_visibility,
            },
            "weights_used": subweights,
            "debug": {
                "source_face_bbox": bbox_src,
                "tryon_face_bbox": bbox_dst,
            },
        }
        return (payload, float(payload["score"]), _json_dumps(payload))


class VTONGarmentScoreNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_image": ("IMAGE", {}),
                "collage_image": ("IMAGE", {}),
                "tryon_image": ("IMAGE", {}),
                "rubric": ("VTON_RUBRIC", {}),
            }
        }

    RETURN_TYPES = ("VTON_SECTION_SCORE", "FLOAT", "STRING")
    RETURN_NAMES = ("garments_section", "garments_score", "garments_json")
    FUNCTION = "score_garments"
    CATEGORY = PACKAGE_CATEGORY

    def _garment_subweights(self, rubric: Dict[str, Any]) -> Dict[str, float]:
        section = rubric["top_level_sections"]["garments_total"]
        return _normalize_subweights(section.get("default_subsections", {}))

    def score_garments(self, source_image: torch.Tensor, collage_image: torch.Tensor, tryon_image: torch.Tensor, rubric: Dict[str, Any]):
        src = _tensor_to_np(source_image)
        collage = _tensor_to_np(collage_image)
        dst = _tensor_to_np(tryon_image)
        section = rubric["top_level_sections"]["garments_total"]
        if not section.get("enabled", True):
            payload = {"enabled": False, "score": 0.0, "confidence": 0.0, "garments": [], "hard_fail": False}
            return (payload, 0.0, _json_dumps(payload))

        parsed_labels = rubric.get("meta", {}).get("parsed_collage_labels") or _parse_collage_labels(collage)
        panels = rubric.get("meta", {}).get("detected_panels") or _split_collage_panels(collage, labels=parsed_labels)
        label_map = _associate_labels_to_panels(panels, parsed_labels)
        src_mask, src_conf = _segment_person(src)
        dst_mask, dst_conf = _segment_person(dst)
        src_bbox = _bbox_from_mask(src_mask) or (0, 0, src.shape[1], src.shape[0])
        dst_bbox = _bbox_from_mask(dst_mask) or (0, 0, dst.shape[1], dst.shape[0])
        subweights = self._garment_subweights(rubric)

        garments_payload = []
        weighted_scores = []
        hard_fail = False
        hard_fail_reasons = []

        for idx, item in enumerate(section.get("inventory", [])):
            if not item.get("enabled", True):
                continue
            panel_index = int(item.get("panel_index", min(idx, max(0, len(panels) - 1))))
            panel_index = max(0, min(panel_index, max(0, len(panels) - 1)))
            panel_bbox = tuple(panels[panel_index]) if panels else (0, 0, collage.shape[1], collage.shape[0])

            label_info = label_map.get(panel_index)
            garment_type = (item.get("garment_type") or "").lower().strip()
            if label_info is not None and label_info.get("primary_label"):
                garment_type = str(label_info.get("primary_label")).lower().strip()

            ref_panel_crop = _crop(collage, panel_bbox)
            ref_crop, ref_mask, ref_object_bbox = _extract_reference_object_crop(ref_panel_crop, garment_type=garment_type)
            if not garment_type or garment_type == "unknown":
                garment_type = _infer_garment_type_heuristic(ref_crop)

            target_bbox = _expected_body_region(dst_bbox, garment_type, dst.shape)
            source_bbox = _expected_body_region(src_bbox, garment_type, src.shape)

            target_region = _crop(dst, target_bbox)
            source_region = _crop(src, source_bbox)

            target_mask = _target_mask_from_region(target_region, garment_type, ref_crop=ref_crop)
            if target_mask.sum() == 0:
                target_mask = _white_bg_foreground_mask(target_region)
            target_mask = _largest_connected_component(target_mask) if target_mask.sum() > 0 else target_mask

            source_mask = _target_mask_from_region(source_region, garment_type, ref_crop=ref_crop)
            if source_mask.sum() == 0:
                source_mask = _white_bg_foreground_mask(source_region)
            source_mask = _largest_connected_component(source_mask) if source_mask.sum() > 0 else source_mask

            target_crop, target_mask_crop, target_object_bbox = _crop_to_mask(target_region, target_mask, pad_scale=0.08)
            source_region_crop, source_mask_crop, source_object_bbox = _crop_to_mask(source_region, source_mask, pad_scale=0.08)

            visibility = _visibility_confidence(target_mask_crop, _bbox_from_mask(target_mask_crop), target_region.shape, extra_quality=_blur_score(target_crop))

            ref_target_color = 0.55 * _masked_hist_similarity_rgb(ref_crop, ref_mask, target_crop, target_mask_crop) + 0.45 * _masked_lab_color_similarity(ref_crop, ref_mask, target_crop, target_mask_crop)
            ref_target_texture = _masked_texture_similarity(ref_crop, ref_mask, target_crop, target_mask_crop)
            ref_text = _ocr_text(ref_crop, psm=7)
            target_text = _ocr_text(target_crop, psm=7)
            meaningful_text = max(len(ref_text.strip()), len(target_text.strip())) >= 3
            text_gate = 1.0 if meaningful_text and (ref_text or target_text) else 0.0
            ref_target_text = _text_similarity(ref_text, target_text) if text_gate > 0 else 1.0

            tc_weights = {
                "color": 0.58,
                "texture": 0.42,
                "text": 0.0,
            }
            if text_gate > 0:
                tc_weights = {
                    "color": 0.46,
                    "texture": 0.34,
                    "text": 0.20,
                }
            ref_target_texture_color_text = (
                tc_weights["color"] * ref_target_color
                + tc_weights["texture"] * ref_target_texture
                + tc_weights["text"] * ref_target_text
            )

            ref_source_color = 0.55 * _masked_hist_similarity_rgb(ref_crop, ref_mask, source_region_crop, source_mask_crop) + 0.45 * _masked_lab_color_similarity(ref_crop, ref_mask, source_region_crop, source_mask_crop)
            ref_source_texture = _masked_texture_similarity(ref_crop, ref_mask, source_region_crop, source_mask_crop)
            ref_source_combo = 0.55 * ref_source_color + 0.45 * ref_source_texture

            shape_structure = (
                0.45 * _best_masked_edge_similarity(ref_crop, ref_mask, target_crop, target_mask_crop, garment_type=garment_type)
                + 0.35 * _mask_aspect_similarity(ref_mask, target_mask_crop)
                + 0.20 * _mask_fill_similarity(ref_mask, target_mask_crop)
            )
            placement = 0.0 if target_mask.sum() == 0 else _clamp01(min(1.0, float(target_mask.mean()) / 0.22))

            sim_ref_target = 0.42 * ref_target_color + 0.26 * ref_target_texture + 0.32 * shape_structure
            sim_ref_source = 0.50 * ref_source_combo + 0.50 * _best_masked_edge_similarity(ref_crop, ref_mask, source_region_crop, source_mask_crop, garment_type=garment_type)
            presence_threshold = _presence_threshold_for_type(garment_type)
            presence = 1.0 if (sim_ref_target >= presence_threshold and sim_ref_target >= sim_ref_source - 0.04 and placement >= 0.10) else 0.0

            missing_reason = None
            leakage_penalty = 0.0
            if presence == 0.0 and bool(item.get("required", True)):
                leakage_penalty = _clamp01(sim_ref_source)
                missing_reason = "required_garment_missing_or_selfie_leakthrough"
                if section.get("hard_fail_on_missing_required", True):
                    hard_fail = True
                    hard_fail_reasons.append(f"{garment_type or item.get('garment_id', 'garment')} missing")

            garment_score = 0.0 if presence == 0.0 else visibility * (
                subweights.get("presence", 0.0) * presence
                + subweights.get("shape_structure", 0.0) * shape_structure
                + subweights.get("texture_color_text", 0.0) * ref_target_texture_color_text
                + subweights.get("placement", 0.0) * placement
            )
            garment_score = _clamp01(max(0.0, garment_score - leakage_penalty * 0.15))

            garment_payload = {
                "garment_id": item.get("garment_id", f"garment_{idx+1}"),
                "garment_type": garment_type,
                "required": bool(item.get("required", True)),
                "detected": bool(presence > 0.0),
                "visibility_confidence": visibility,
                "panel_index": panel_index,
                "scores": {
                    "presence": presence,
                    "shape_structure": shape_structure,
                    "texture_color_text": ref_target_texture_color_text,
                    "placement": placement,
                },
                "reference_vs_source_debug": {
                    "reference_target_similarity": sim_ref_target,
                    "reference_source_similarity": sim_ref_source,
                    "leakage_penalty": leakage_penalty,
                    "presence_threshold": presence_threshold,
                    "ref_object_bbox_within_panel": ref_object_bbox,
                    "target_object_bbox_within_region": target_object_bbox,
                    "source_object_bbox_within_region": source_object_bbox,
                    "collage_label": label_info.get("primary_label") if label_info else item.get("garment_type", ""),
                    "ref_text": ref_text,
                    "target_text": target_text,
                    "text_gate": text_gate,
                },
                "garment_score": garment_score,
                "failure_reasons": [missing_reason] if missing_reason else [],
                "weight_within_garments": float(item.get("normalized_weight_within_garments", 0.0)),
                "label_source": item.get("label_source", "rubric_or_inferred"),
            }
            garments_payload.append(garment_payload)
            weighted_scores.append(garment_score * garment_payload["weight_within_garments"])

        garments_score = 0.0 if hard_fail else float(sum(weighted_scores))
        confidence = float(np.mean([g["visibility_confidence"] for g in garments_payload])) if garments_payload else min(src_conf, dst_conf)
        payload = {
            "enabled": True,
            "score": garments_score,
            "confidence": confidence,
            "hard_fail": hard_fail,
            "hard_fail_reasons": hard_fail_reasons,
            "weights_used": subweights,
            "garments": garments_payload,
            "debug": {
                "panel_count": len(panels),
                "label_count": len(parsed_labels),
                "parsed_labels": parsed_labels,
                "src_person_bbox": src_bbox,
                "dst_person_bbox": dst_bbox,
            },
        }
        return (payload, float(payload["score"]), _json_dumps(payload))



class VTONBodyShapeScoreNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_image": ("IMAGE", {}),
                "tryon_image": ("IMAGE", {}),
                "rubric": ("VTON_RUBRIC", {}),
            }
        }

    RETURN_TYPES = ("VTON_SECTION_SCORE", "FLOAT", "STRING")
    RETURN_NAMES = ("body_section", "body_score", "body_json")
    FUNCTION = "score_body"
    CATEGORY = PACKAGE_CATEGORY

    def score_body(self, source_image: torch.Tensor, tryon_image: torch.Tensor, rubric: Dict[str, Any]):
        src = _tensor_to_np(source_image)
        dst = _tensor_to_np(tryon_image)
        section = rubric["top_level_sections"]["body_shape"]
        if not section.get("enabled", True):
            payload = {"enabled": False, "score": 0.0, "confidence": 0.0, "subscores": {}}
            return (payload, 0.0, _json_dumps(payload))
        subweights = _normalize_subweights(section.get("subsections", {}))
        src_mask, src_conf = _segment_person(src)
        dst_mask, dst_conf = _segment_person(dst)
        ratio_similarity = _body_ratio_similarity(src_mask, dst_mask)
        silhouette_similarity = _silhouette_similarity(src_mask, dst_mask)
        confidence = _clamp01((src_conf + dst_conf) / 2.0)
        score = confidence * (
            subweights.get("ratio_similarity", 0.0) * ratio_similarity
            + subweights.get("silhouette_similarity", 0.0) * silhouette_similarity
        )
        payload = {
            "enabled": True,
            "score": score,
            "confidence": confidence,
            "subscores": {
                "ratio_similarity": ratio_similarity,
                "silhouette_similarity": silhouette_similarity,
            },
            "weights_used": subweights,
        }
        return (payload, float(payload["score"]), _json_dumps(payload))


class VTONLightingScoreNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tryon_image": ("IMAGE", {}),
                "rubric": ("VTON_RUBRIC", {}),
            }
        }

    RETURN_TYPES = ("VTON_SECTION_SCORE", "FLOAT", "STRING")
    RETURN_NAMES = ("lighting_section", "lighting_score", "lighting_json")
    FUNCTION = "score_lighting"
    CATEGORY = PACKAGE_CATEGORY

    def score_lighting(self, tryon_image: torch.Tensor, rubric: Dict[str, Any]):
        img = _tensor_to_np(tryon_image)
        section = rubric["top_level_sections"]["lighting"]
        if not section.get("enabled", True):
            payload = {"enabled": False, "score": 0.0, "confidence": 0.0, "subscores": {}}
            return (payload, 0.0, _json_dumps(payload))

        subweights = _normalize_subweights(section.get("subsections", {}))
        person_mask, person_conf = _segment_person(img)
        predicted_class, features = _lighting_class(img)
        desired = str(section.get("desired_class", "") or "")
        class_match = _lighting_class_similarity(predicted_class, desired)
        consistency = _lighting_consistency(img, person_mask)
        confidence = _clamp01(0.6 * person_conf + 0.4 * features.get("clarity", 0.5))
        score = confidence * (
            subweights.get("lighting_class_match", 0.0) * class_match
            + subweights.get("photometric_consistency", 0.0) * consistency
        )
        payload = {
            "enabled": True,
            "score": score,
            "confidence": confidence,
            "predicted_class": predicted_class,
            "desired_class": desired,
            "subscores": {
                "lighting_class_match": class_match,
                "photometric_consistency": consistency,
            },
            "weights_used": subweights,
            "debug_features": features,
        }
        return (payload, float(payload["score"]), _json_dumps(payload))


class VTONBackgroundScoreNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tryon_image": ("IMAGE", {}),
                "rubric": ("VTON_RUBRIC", {}),
            }
        }

    RETURN_TYPES = ("VTON_SECTION_SCORE", "FLOAT", "STRING")
    RETURN_NAMES = ("background_section", "background_score", "background_json")
    FUNCTION = "score_background"
    CATEGORY = PACKAGE_CATEGORY

    def score_background(self, tryon_image: torch.Tensor, rubric: Dict[str, Any]):
        img = _tensor_to_np(tryon_image)
        section = rubric["top_level_sections"]["background"]
        if not section.get("enabled", True):
            payload = {"enabled": False, "score": 0.0, "confidence": 0.0, "subscores": {}}
            return (payload, 0.0, _json_dumps(payload))

        subweights = _normalize_subweights(section.get("subsections", {}))
        person_mask, person_conf = _segment_person(img)
        subs = _background_scores(img, person_mask)
        confidence = person_conf
        score = confidence * sum(subweights.get(k, 0.0) * subs.get(k, 0.0) for k in subweights.keys())
        payload = {
            "enabled": True,
            "score": score,
            "confidence": confidence,
            "subscores": subs,
            "weights_used": subweights,
        }
        return (payload, float(payload["score"]), _json_dumps(payload))


class VTONAggregateRubricNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "rubric": ("VTON_RUBRIC", {}),
                "face_section": ("VTON_SECTION_SCORE", {}),
                "garments_section": ("VTON_SECTION_SCORE", {}),
                "body_section": ("VTON_SECTION_SCORE", {}),
                "lighting_section": ("VTON_SECTION_SCORE", {}),
                "background_section": ("VTON_SECTION_SCORE", {}),
            }
        }

    RETURN_TYPES = ("FLOAT", "STRING", "STRING")
    RETURN_NAMES = ("final_score", "report_json", "summary_text")
    FUNCTION = "aggregate"
    CATEGORY = PACKAGE_CATEGORY
    OUTPUT_NODE = True

    def aggregate(self, rubric: Dict[str, Any], face_section: Dict[str, Any], garments_section: Dict[str, Any], body_section: Dict[str, Any], lighting_section: Dict[str, Any], background_section: Dict[str, Any]):
        report = _report_shell()
        sections = rubric["top_level_sections"]
        top_section_map = {
            "face_identity": face_section,
            "garments_total": garments_section,
            "body_shape": body_section,
            "lighting": lighting_section,
            "background": background_section,
        }
        weights_used = _normalize_enabled_weights(sections)
        report["top_level_weights_used"] = weights_used

        hard_fail = bool(garments_section.get("hard_fail", False))
        report["hard_fail"] = hard_fail
        report["hard_fail_reasons"] = list(garments_section.get("hard_fail_reasons", []))
        report["garments"] = garments_section.get("garments", [])

        weighted_sum = 0.0
        for key, section_payload in top_section_map.items():
            report["top_level_scores"][key] = float(section_payload.get("score", 0.0))
            report["top_level_confidence"][key] = float(section_payload.get("confidence", 0.0))
            weighted_sum += weights_used.get(key, 0.0) * float(section_payload.get("score", 0.0))

        final_score = 0.0 if hard_fail else _clamp01(weighted_sum)
        report["final_score"] = final_score
        report["meta"] = {
            "rubric_version": rubric.get("version", "1.0"),
            "speed_profile": rubric.get("options", {}).get("speed_profile", "balanced"),
        }
        if final_score >= 0.85:
            report["operator_recommendation"] = "accept"
        elif final_score >= 0.70:
            report["operator_recommendation"] = "review"
        else:
            report["operator_recommendation"] = "reject"

        summary = (
            f"Final VTON QC score: {final_score:.4f}\n"
            f"Recommendation: {report['operator_recommendation']}\n"
            f"Hard fail: {report['hard_fail']}\n"
            f"Face: {report['top_level_scores'].get('face_identity', 0.0):.4f}\n"
            f"Garments: {report['top_level_scores'].get('garments_total', 0.0):.4f}\n"
            f"Body: {report['top_level_scores'].get('body_shape', 0.0):.4f}\n"
            f"Lighting: {report['top_level_scores'].get('lighting', 0.0):.4f}\n"
            f"Background: {report['top_level_scores'].get('background', 0.0):.4f}"
        )
        return (float(final_score), _json_dumps(report), summary)


class VTONSaveJSONReportNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "report_json": ("STRING", {"multiline": True, "default": ""}),
                "filename_prefix": ("STRING", {"multiline": False, "default": "vton_qc_report"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("saved_path",)
    FUNCTION = "save"
    CATEGORY = PACKAGE_CATEGORY
    OUTPUT_NODE = True

    def save(self, report_json: str, filename_prefix: str):
        output_dir = _get_output_dir()
        filename = f"{filename_prefix}_{uuid.uuid4().hex[:8]}.json"
        path = os.path.join(output_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(report_json)
        return (path,)


NODE_CLASS_MAPPINGS = {
    "VTONRubricLoadNode": VTONRubricLoadNode,
    "VTONFaceIdentityScoreNode": VTONFaceIdentityScoreNode,
    "VTONGarmentScoreNode": VTONGarmentScoreNode,
    "VTONBodyShapeScoreNode": VTONBodyShapeScoreNode,
    "VTONLightingScoreNode": VTONLightingScoreNode,
    "VTONBackgroundScoreNode": VTONBackgroundScoreNode,
    "VTONAggregateRubricNode": VTONAggregateRubricNode,
    "VTONSaveJSONReportNode": VTONSaveJSONReportNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VTONRubricLoadNode": "VTON QC - Load Rubric",
    "VTONFaceIdentityScoreNode": "VTON QC - Face Identity",
    "VTONGarmentScoreNode": "VTON QC - Garments",
    "VTONBodyShapeScoreNode": "VTON QC - Body Shape",
    "VTONLightingScoreNode": "VTON QC - Lighting",
    "VTONBackgroundScoreNode": "VTON QC - Background",
    "VTONAggregateRubricNode": "VTON QC - Aggregate",
    "VTONSaveJSONReportNode": "VTON QC - Save JSON",
}

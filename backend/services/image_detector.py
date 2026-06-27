from __future__ import annotations

from dataclasses import dataclass

import cv2
import fitz
import numpy as np

from backend.services.pdf_reader import render_page


@dataclass(frozen=True)
class ImageDetectionResult:
    count: int
    confidence: float
    rects: list[tuple[float, float, float, float]]


def detect_photos(page: fitz.Page, page_rect, scanned: bool) -> ImageDetectionResult:
    rects = detect_photo_regions(page, page_rect, scanned)
    if not rects:
        return ImageDetectionResult(count=0, confidence=0.0, rects=[])

    embedded = _embedded_photo_rects(page, page_rect)
    contour_rects = _contour_photo_rects(page) if scanned or not embedded else []
    confidence = 0.82 if embedded else 0.68
    if embedded and contour_rects:
        confidence = 0.88
    return ImageDetectionResult(count=len(rects), confidence=confidence, rects=rects)


def detect_photo_regions(page: fitz.Page, page_rect, scanned: bool) -> list[tuple[float, float, float, float]]:
    embedded = _embedded_photo_rects(page, page_rect)
    contour_rects = _contour_photo_rects(page) if scanned or not embedded else []
    return _dedupe_rects([*embedded, *contour_rects])


def _embedded_photo_rects(page: fitz.Page, page_rect) -> list[tuple[float, float, float, float]]:
    page_area = float(page_rect.width * page_rect.height)
    rects: list[tuple[float, float, float, float]] = []

    for image in page.get_images(full=True):
        xref = image[0]
        for rect in page.get_image_rects(xref):
            w, h = float(rect.width), float(rect.height)
            area = w * h
            if _is_photo_like(w, h, area, page_area, rect.y0, page_rect.height):
                rects.append((float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1)))
    return rects


def _contour_photo_rects(page: fitz.Page) -> list[tuple[float, float, float, float]]:
    image = render_page(page, dpi=135)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    page_h, page_w = gray.shape[:2]
    page_area = page_w * page_h

    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, 70, 170)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (19, 19))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rects: list[tuple[float, float, float, float]] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if not _is_photo_like(w, h, area, page_area, y, page_h):
            continue

        roi = gray[y : y + h, x : x + w]
        if roi.size == 0:
            continue
        texture = float(np.std(roi))
        edge_density = float(np.count_nonzero(edges[y : y + h, x : x + w])) / max(area, 1)
        if texture < 18 or edge_density < 0.015:
            continue

        scale_x = float(page.rect.width) / page_w
        scale_y = float(page.rect.height) / page_h
        rects.append((x * scale_x, y * scale_y, (x + w) * scale_x, (y + h) * scale_y))

    return rects


def _is_photo_like(
    width: float,
    height: float,
    area: float,
    page_area: float,
    y0: float,
    page_height: float,
) -> bool:
    if width < 55 or height < 45:
        return False
    if area < page_area * 0.012 or area > page_area * 0.72:
        return False
    aspect = width / max(height, 1.0)
    if aspect < 0.28 or aspect > 4.8:
        return False
    if y0 < page_height * 0.035 and area < page_area * 0.05:
        return False
    return True


def _dedupe_rects(rects: list[tuple[float, float, float, float]]) -> list[tuple[float, float, float, float]]:
    deduped: list[tuple[float, float, float, float]] = []
    for rect in sorted(rects, key=_area, reverse=True):
        if all(_iou(rect, existing) < 0.45 for existing in deduped):
            deduped.append(rect)
    return deduped


def _area(rect: tuple[float, float, float, float]) -> float:
    return max(0.0, rect[2] - rect[0]) * max(0.0, rect[3] - rect[1])


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    x0 = max(a[0], b[0])
    y0 = max(a[1], b[1])
    x1 = min(a[2], b[2])
    y1 = min(a[3], b[3])
    intersection = _area((x0, y0, x1, y1))
    union = _area(a) + _area(b) - intersection
    return intersection / union if union else 0.0

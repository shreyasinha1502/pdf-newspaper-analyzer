from __future__ import annotations

import threading
from dataclasses import dataclass

import cv2
import numpy as np

from backend.utils.config import settings


@dataclass(frozen=True)
class OCRTextBox:
    text: str
    bbox: tuple[float, float, float, float]
    font_size: float
    bold: bool = False

    @property
    def width(self) -> float:
        return max(0.0, self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> float:
        return max(0.0, self.bbox[3] - self.bbox[1])


class OCREngine:
    def __init__(self) -> None:
        self._paddle = None
        self._paddle_failed = False
        self._lock = threading.Lock()

    def read(self, image: np.ndarray) -> list[OCRTextBox]:
        with self._lock:
            boxes = self._read_with_paddle(image)
            if boxes:
                return boxes
            return self._read_with_tesseract(image)

    def _read_with_paddle(self, image: np.ndarray) -> list[OCRTextBox]:
        if self._paddle_failed:
            return []
        try:
            if self._paddle is None:
                from paddleocr import PaddleOCR

                self._paddle = PaddleOCR(
                    use_angle_cls=True,
                    lang=settings.paddleocr_language,
                    show_log=False,
                )

            result = self._paddle.ocr(image, cls=True)
            boxes: list[OCRTextBox] = []
            for page_result in result or []:
                for item in page_result or []:
                    points, payload = item
                    text, confidence = payload
                    if confidence < 0.45 or not str(text).strip():
                        continue
                    xs = [float(point[0]) for point in points]
                    ys = [float(point[1]) for point in points]
                    y0, y1 = min(ys), max(ys)
                    boxes.append(
                        OCRTextBox(
                            text=str(text).strip(),
                            bbox=(min(xs), y0, max(xs), y1),
                            font_size=max(1.0, y1 - y0),
                            bold=False,
                        )
                    )
            return _merge_ocr_words(boxes)
        except Exception:
            self._paddle_failed = True
            return []

    def _read_with_tesseract(self, image: np.ndarray) -> list[OCRTextBox]:
        try:
            import pytesseract

            rgb = image
            data = pytesseract.image_to_data(
                rgb,
                output_type=pytesseract.Output.DICT,
                lang=settings.ocr_languages,
                config="--psm 6",
            )
        except Exception:
            return []

        boxes: list[OCRTextBox] = []
        count = len(data.get("text", []))
        for index in range(count):
            text = str(data["text"][index]).strip()
            confidence = float(data["conf"][index]) if str(data["conf"][index]).replace(".", "", 1).isdigit() else -1
            if not text or confidence < 45:
                continue
            x = float(data["left"][index])
            y = float(data["top"][index])
            w = float(data["width"][index])
            h = float(data["height"][index])
            boxes.append(OCRTextBox(text=text, bbox=(x, y, x + w, y + h), font_size=max(1.0, h)))

        return _merge_ocr_words(boxes)


def get_ocr_engine() -> OCREngine:
    global _ENGINE
    try:
        return _ENGINE
    except NameError:
        _ENGINE = OCREngine()
        return _ENGINE


def _merge_ocr_words(words: list[OCRTextBox]) -> list[OCRTextBox]:
    if not words:
        return []

    rows: list[list[OCRTextBox]] = []
    for word in sorted(words, key=lambda item: (item.bbox[1], item.bbox[0])):
        matched = False
        for row in rows:
            row_mid = (row[0].bbox[1] + row[0].bbox[3]) / 2
            word_mid = (word.bbox[1] + word.bbox[3]) / 2
            if abs(row_mid - word_mid) <= max(row[0].height, word.height) * 0.6:
                row.append(word)
                matched = True
                break
        if not matched:
            rows.append([word])

    merged: list[OCRTextBox] = []
    for row in rows:
        row = sorted(row, key=lambda item: item.bbox[0])
        pieces: list[list[OCRTextBox]] = [[row[0]]]
        for word in row[1:]:
            gap = word.bbox[0] - pieces[-1][-1].bbox[2]
            if gap <= max(word.height * 1.8, 22):
                pieces[-1].append(word)
            else:
                pieces.append([word])

        for piece in pieces:
            text = " ".join(item.text for item in piece)
            x0 = min(item.bbox[0] for item in piece)
            y0 = min(item.bbox[1] for item in piece)
            x1 = max(item.bbox[2] for item in piece)
            y1 = max(item.bbox[3] for item in piece)
            merged.append(OCRTextBox(text=text, bbox=(x0, y0, x1, y1), font_size=y1 - y0))

    return merged

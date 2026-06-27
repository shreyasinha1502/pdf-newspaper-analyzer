from __future__ import annotations

from dataclasses import dataclass

import fitz
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class TextLine:
    text: str
    bbox: tuple[float, float, float, float]
    font_size: float
    bold: bool

    @property
    def width(self) -> float:
        return max(0.0, self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> float:
        return max(0.0, self.bbox[3] - self.bbox[1])


def page_has_selectable_text(page: fitz.Page) -> bool:
    text = page.get_text("text").strip()
    return len(text) >= 40


def extract_text_lines(page: fitz.Page) -> list[TextLine]:
    raw = page.get_text("rawdict")
    lines: list[TextLine] = []

    for block in raw.get("blocks", []):
        if block.get("type") != 0:
            continue
        for raw_line in block.get("lines", []):
            spans = raw_line.get("spans", [])
            pieces = []
            sizes = []
            bold = False
            boxes = []

            for span in spans:
                text = _span_text(span)
                if not text.strip():
                    continue
                pieces.append(text)
                sizes.append(float(span.get("size", 0.0)))
                font_name = str(span.get("font", "")).lower()
                flags = int(span.get("flags", 0))
                bold = bold or "bold" in font_name or bool(flags & 16)
                boxes.append(tuple(float(v) for v in span.get("bbox", (0, 0, 0, 0))))

            line_text = " ".join("".join(pieces).split())
            if not line_text or not boxes:
                continue

            x0 = min(box[0] for box in boxes)
            y0 = min(box[1] for box in boxes)
            x1 = max(box[2] for box in boxes)
            y1 = max(box[3] for box in boxes)
            lines.append(
                TextLine(
                    text=line_text,
                    bbox=(x0, y0, x1, y1),
                    font_size=max(sizes) if sizes else y1 - y0,
                    bold=bold,
                )
            )

    return lines


def render_page(page: fitz.Page, dpi: int = 180) -> np.ndarray:
    matrix = fitz.Matrix(dpi / 72, dpi / 72)
    pixmap = page.get_pixmap(matrix=matrix, alpha=False)
    image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
    return np.asarray(image)


def _span_text(span: dict) -> str:
    if "text" in span:
        return str(span["text"])
    chars = span.get("chars", [])
    return "".join(char.get("c", "") for char in chars)

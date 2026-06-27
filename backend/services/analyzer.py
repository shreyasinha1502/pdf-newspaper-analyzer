from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import fitz

from backend.services.headline_detector import detect_headlines
from backend.services.image_detector import detect_photos
from backend.services.ocr import OCRTextBox, get_ocr_engine
from backend.services.pdf_reader import extract_text_lines, page_has_selectable_text, render_page
from backend.utils.config import settings
from backend.utils.errors import AnalysisError
from backend.utils.logging import get_logger

logger = get_logger(__name__)


class PDFAnalyzer:
    def analyze(self, pdf_path: Path) -> dict:
        try:
            with fitz.open(pdf_path) as document:
                if document.needs_pass:
                    raise AnalysisError("Password protected PDFs are not supported.")
                pages = document.page_count
        except fitz.FileDataError as exc:
            raise AnalysisError("The uploaded PDF is corrupted or invalid.") from exc

        if pages == 0:
            raise AnalysisError("The uploaded PDF contains no pages.")
        if pages > settings.max_pages:
            raise AnalysisError(f"PDF has {pages} pages; the limit is {settings.max_pages}.")

        totals = {
            "pages": pages,
            "headlines": 0,
            "photos": 0,
            "headline_confidence": 0.0,
            "photo_confidence": 0.0,
            "logs": [],
        }

        worker_count = min(settings.max_workers, pages)
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(_analyze_page, pdf_path, page_number): page_number
                for page_number in range(pages)
            }
            for future in as_completed(futures):
                page_result = future.result()
                totals["headlines"] += page_result["headlines"]
                totals["photos"] += page_result["photos"]
                totals["headline_confidence"] += page_result["headline_confidence"]
                totals["photo_confidence"] += page_result["photo_confidence"]
                totals["logs"].append(page_result["log"])

        totals["headline_confidence"] = round(totals["headline_confidence"] / pages, 3)
        totals["photo_confidence"] = round(totals["photo_confidence"] / pages, 3)
        totals["logs"] = sorted(totals["logs"], key=lambda item: item["page"])
        return totals


def _analyze_page(pdf_path: Path, page_index: int) -> dict:
    with fitz.open(pdf_path) as document:
        page = document.load_page(page_index)
        page_rect = page.rect
        has_text = page_has_selectable_text(page)

        if has_text:
            lines = extract_text_lines(page)
            mode = "digital"
        else:
            image = render_page(page, dpi=settings.ocr_dpi)
            lines = _scale_ocr_lines(_ocr_lines(image), image.shape, page_rect)
            mode = "ocr"

        photo_result = detect_photos(page, page_rect, scanned=not has_text)
        headline_result = detect_headlines(lines, page_rect, image_rects=photo_result.rects)

        logger.info(
            "page=%s mode=%s headlines=%s photos=%s",
            page_index + 1,
            mode,
            headline_result.count,
            photo_result.count,
        )

        return {
            "headlines": headline_result.count,
            "photos": photo_result.count,
            "headline_confidence": headline_result.confidence,
            "photo_confidence": photo_result.confidence,
            "log": {
                "page": page_index + 1,
                "mode": mode,
                "headlines": headline_result.count,
                "photos": photo_result.count,
            },
        }


def _ocr_lines(image) -> list[OCRTextBox]:
    try:
        return get_ocr_engine().read(image)
    except Exception as exc:
        logger.warning("OCR failed: %s", exc)
        return []


def _scale_ocr_lines(
    lines: list[OCRTextBox],
    image_shape,
    page_rect,
) -> list[OCRTextBox]:
    image_height, image_width = image_shape[:2]
    scale_x = float(page_rect.width) / max(float(image_width), 1.0)
    scale_y = float(page_rect.height) / max(float(image_height), 1.0)

    scaled: list[OCRTextBox] = []
    for line in lines:
        x0, y0, x1, y1 = line.bbox
        scaled.append(
            OCRTextBox(
                text=line.text,
                bbox=(x0 * scale_x, y0 * scale_y, x1 * scale_x, y1 * scale_y),
                font_size=line.font_size * scale_y,
                bold=line.bold,
            )
        )
    return scaled

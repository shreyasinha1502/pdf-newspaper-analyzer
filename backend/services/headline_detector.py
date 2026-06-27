from __future__ import annotations

import re
import statistics
import unicodedata
from dataclasses import dataclass
from typing import Protocol


class LineLike(Protocol):
    text: str
    bbox: tuple[float, float, float, float]
    font_size: float
    bold: bool

    @property
    def width(self) -> float: ...

    @property
    def height(self) -> float: ...


@dataclass(frozen=True)
class DetectionResult:
    count: int
    confidence: float


@dataclass
class Candidate:
    text: str
    bbox: tuple[float, float, float, float]
    score: float
    font_size: float
    line_height: float


NOISE_WORDS = {
    "weather",
    "classified",
    "advertisement",
    "advertorial",
    "sponsored",
    "date",
    "edition",
    "page",
    "continued",
    "photo",
    "caption",
    "courtesy",
    "subscribe",
    "www.",
    "http",
    "@",
    "फोटो",
    "चित्र",
    "तस्वीर",
    "साभार",
    "विज्ञापन",
    "संवाददाता",
    "प्रतिनिधि",
    "एजेंसी",
    "पेज",
}

NOISE_PREFIXES = (
    "by ",
    "from ",
    "photo by",
    "image by",
    "source:",
    "caption:",
    "continued from",
    "फोटो",
    "चित्र",
    "तस्वीर",
    "साभार",
    "संवाददाता",
    "प्रतिनिधि",
)


def detect_headlines(
    lines: list[LineLike],
    page_rect,
    image_rects: list[tuple[float, float, float, float]] | None = None,
) -> DetectionResult:
    clean_lines = [line for line in lines if _valid_text(line.text)]
    if not clean_lines:
        return DetectionResult(count=0, confidence=0.0)

    body_size = _estimate_body_size(clean_lines)
    page_width = float(page_rect.width)
    page_height = float(page_rect.height)

    candidates: list[Candidate] = []
    for line in clean_lines:
        score = _headline_score(
            line,
            body_size,
            page_width,
            page_height,
            image_rects or [],
        )
        if score + 1e-9 >= _candidate_threshold(line, body_size):
            candidates.append(Candidate(line.text, line.bbox, score, line.font_size, line.height))

    merged = _merge_subheadlines(_dedupe_candidates(_merge_broken_headlines(candidates)))
    if not merged:
        return DetectionResult(count=0, confidence=0.0)

    confidence = sum(item.score for item in merged) / len(merged)
    return DetectionResult(count=len(merged), confidence=round(min(confidence, 0.99), 3))


def _estimate_body_size(lines: list[LineLike]) -> float:
    sizes = sorted(line.font_size for line in lines if 5 <= line.font_size <= 24)
    if not sizes:
        return 10.0
    lower_sample_size = max(1, int(len(sizes) * 0.35))
    return statistics.median(sizes[:lower_sample_size])


def _headline_score(
    line: LineLike,
    body_size: float,
    page_width: float,
    page_height: float,
    image_rects: list[tuple[float, float, float, float]],
) -> float:
    text = line.text.strip()
    lowered = text.lower()
    words = _tokens(text)
    if len(words) < 2 or len(words) > 18 or len(text) > 135:
        return 0.0

    y0, y1 = line.bbox[1], line.bbox[3]
    if y0 < page_height * 0.045 or y1 > page_height * 0.955:
        return 0.0
    if lowered.startswith(NOISE_PREFIXES):
        return 0.0
    if any(word in lowered for word in NOISE_WORDS):
        return 0.0
    if re.fullmatch(r"[\d\s|:/.-]+", text):
        return 0.0
    if _looks_like_metadata(text):
        return 0.0
    if _looks_like_caption(line, body_size, image_rects):
        return 0.0

    size_ratio = line.font_size / max(body_size, 1.0)
    width_ratio = line.width / max(page_width, 1.0)
    uppercase_ratio = _uppercase_ratio(text)
    title_ratio = _titlecase_ratio(text)
    indic_ratio = _indic_letter_ratio(text)

    clear_size_hierarchy = line.font_size >= body_size + 2.6 or size_ratio >= 1.3
    bold_headline = line.bold and (line.font_size >= body_size + 1.7 or size_ratio >= 1.18)
    compact_prominent = (
        2 <= len(words) <= 7
        and not text.endswith((".", "।", ":", ";"))
        and (line.font_size >= body_size + 1.6 or size_ratio >= 1.16)
    )
    ocr_large_text = not line.bold and size_ratio >= 1.42
    if not (clear_size_hierarchy or bold_headline or compact_prominent or ocr_large_text):
        return 0.0

    score = 0.0
    if size_ratio >= 1.65:
        score += 0.48
    elif size_ratio >= 1.42:
        score += 0.40
    elif size_ratio >= 1.25:
        score += 0.30
    elif compact_prominent:
        score += 0.20
    elif line.bold:
        score += 0.20

    if line.bold:
        score += 0.15
    if 0.10 <= width_ratio <= 0.86:
        score += 0.10
    if uppercase_ratio >= 0.58:
        score += 0.10
    elif title_ratio >= 0.42:
        score += 0.07
    elif indic_ratio >= 0.45:
        score += 0.07
    if 3 <= len(words) <= 14:
        score += 0.08
    if not text.endswith((".", ":", ";")):
        score += 0.05
    if line.height >= body_size * 1.15:
        score += 0.06
    if _looks_like_body_sentence(text):
        score -= 0.18
    if width_ratio < 0.18 and len(words) <= 3:
        score -= 0.12
    if compact_prominent and width_ratio < 0.12:
        score -= 0.08

    return max(0.0, min(score, 0.99))


def _candidate_threshold(line: LineLike, body_size: float) -> float:
    tokens = _tokens(line.text)
    compact_bold_headline = (
        line.bold
        and 2 <= len(tokens) <= 7
        and not line.text.strip().endswith((".", "।", ":", ";"))
        and (line.font_size >= body_size + 1.35 or line.font_size / max(body_size, 1.0) >= 1.14)
    )
    compact_indic_headline = (
        line.bold
        and _indic_letter_ratio(line.text) >= 0.45
        and 2 <= len(tokens) <= 6
        and (line.font_size >= body_size + 1.2 or line.font_size / max(body_size, 1.0) >= 1.12)
    )
    if compact_indic_headline:
        return 0.58
    if compact_bold_headline:
        return 0.60
    compact_prominent_headline = (
        2 <= len(tokens) <= 7
        and not line.text.strip().endswith((".", "।", ":", ";"))
        and (line.font_size >= body_size + 2.0 or line.font_size / max(body_size, 1.0) >= 1.22)
    )
    if compact_prominent_headline:
        return 0.64
    return 0.68


def _merge_broken_headlines(candidates: list[Candidate]) -> list[Candidate]:
    ordered = sorted(candidates, key=lambda item: (item.bbox[1], item.bbox[0]))
    merged: list[Candidate] = []

    for item in ordered:
        if not merged:
            merged.append(item)
            continue

        previous = merged[-1]
        vertical_gap = item.bbox[1] - previous.bbox[3]
        same_column = _same_column(previous.bbox, item.bbox)
        compatible_width = (
            abs((item.bbox[2] - item.bbox[0]) - (previous.bbox[2] - previous.bbox[0])) < 180
            or _horizontal_overlap_ratio(previous.bbox, item.bbox) > 0.62
        )

        if 0 <= vertical_gap <= max(18, previous.line_height * 0.75) and same_column and compatible_width:
            merged[-1] = Candidate(
                text=f"{previous.text} {item.text}",
                bbox=(
                    min(previous.bbox[0], item.bbox[0]),
                    previous.bbox[1],
                    max(previous.bbox[2], item.bbox[2]),
                    item.bbox[3],
                ),
                score=max(previous.score, item.score),
                font_size=max(previous.font_size, item.font_size),
                line_height=max(previous.line_height, item.line_height),
            )
        else:
            merged.append(item)

    return merged


def _merge_subheadlines(candidates: list[Candidate]) -> list[Candidate]:
    ordered = sorted(candidates, key=lambda item: (item.bbox[1], item.bbox[0]))
    merged: list[Candidate] = []

    for item in ordered:
        if not merged:
            merged.append(item)
            continue

        previous = merged[-1]
        vertical_gap = item.bbox[1] - previous.bbox[3]
        if (
            0 <= vertical_gap <= max(30, previous.line_height * 1.15)
            and _same_column(previous.bbox, item.bbox)
            and _horizontal_overlap_ratio(previous.bbox, item.bbox) >= 0.35
            and item.font_size <= previous.font_size * 0.9
            and item.score <= previous.score + 0.08
        ):
            merged[-1] = Candidate(
                text=f"{previous.text} {item.text}",
                bbox=(
                    min(previous.bbox[0], item.bbox[0]),
                    previous.bbox[1],
                    max(previous.bbox[2], item.bbox[2]),
                    item.bbox[3],
                ),
                score=max(previous.score, item.score),
                font_size=previous.font_size,
                line_height=max(previous.line_height, item.line_height),
            )
        else:
            merged.append(item)

    return merged


def _dedupe_candidates(candidates: list[Candidate]) -> list[Candidate]:
    deduped: list[Candidate] = []
    for item in sorted(candidates, key=lambda candidate: candidate.score, reverse=True):
        normalized = _normalize_text(item.text)
        duplicate = False
        for existing in deduped:
            if _iou(item.bbox, existing.bbox) > 0.35:
                duplicate = True
                break
            existing_normalized = _normalize_text(existing.text)
            if normalized and normalized == existing_normalized:
                duplicate = True
                break
        if not duplicate:
            deduped.append(item)

    return sorted(deduped, key=lambda candidate: (candidate.bbox[1], candidate.bbox[0]))


def _valid_text(text: str) -> bool:
    text = text.strip()
    if len(text) < 4:
        return False
    words = _tokens(text)
    if len(words) == 1 and len(text) < 9:
        return False
    return _has_letter(text)


def _looks_like_metadata(text: str) -> bool:
    lowered = text.lower().strip()
    if re.search(r"\b(mon|tue|wed|thu|fri|sat|sun)(day)?\b", lowered):
        return True
    month_pattern = (
        r"\b(jan\.?|january|feb\.?|february|mar\.?|march|apr\.?|april|may|"
        r"jun\.?|june|jul\.?|july|aug\.?|august|sep\.?|sept\.?|september|"
        r"oct\.?|october|nov\.?|november|dec\.?|december)\b"
    )
    if re.search(month_pattern, lowered):
        return True
    if re.search(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", lowered):
        return True
    if re.search(r"\b(rs\.?|₹|\$)\s?\d+", lowered):
        return True
    if re.search(r"\b\d+\s?(am|pm)\b", lowered):
        return True
    return False


def _looks_like_body_sentence(text: str) -> bool:
    words = _tokens(text)
    if len(words) >= 12 and text.endswith((".", "।")):
        return True
    if len(words) >= 10 and ("," in text or "،" in text) and _uppercase_ratio(text) < 0.3:
        return True
    return False


def _looks_like_caption(
    line: LineLike,
    body_size: float,
    image_rects: list[tuple[float, float, float, float]],
) -> bool:
    if not image_rects:
        return False

    text = line.text.strip().lower()
    if text.startswith(("caption", "photo", "image", "courtesy", "source")):
        return True

    for image_rect in image_rects:
        horizontal_overlap = _horizontal_overlap_ratio(line.bbox, image_rect)
        below_image_gap = line.bbox[1] - image_rect[3]
        inside_image = _center_inside(line.bbox, image_rect)
        font_is_caption_sized = line.font_size <= body_size * 1.35
        line_is_not_dominant = line.width <= (image_rect[2] - image_rect[0]) * 1.25

        if horizontal_overlap >= 0.35 and 0 <= below_image_gap <= max(42, body_size * 3.5):
            if font_is_caption_sized or line_is_not_dominant:
                return True
        if inside_image and font_is_caption_sized:
            return True

    return False


def _uppercase_ratio(text: str) -> float:
    letters = [char for char in text if char.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for char in letters if char.isupper()) / len(letters)


def _titlecase_ratio(text: str) -> float:
    words = [word for word in _tokens(text) if _has_letter(word)]
    if not words:
        return 0.0
    title_words = [word for word in words if word[:1].isupper()]
    return len(title_words) / len(words)


def _normalize_text(text: str) -> str:
    normalized = []
    previous_space = False
    for char in text.lower():
        if char.isalnum() or unicodedata.category(char).startswith("M"):
            normalized.append(char)
            previous_space = False
        elif not previous_space:
            normalized.append(" ")
            previous_space = True
    return "".join(normalized).strip()


def _tokens(text: str) -> list[str]:
    tokens: list[str] = []
    current: list[str] = []
    for char in text:
        category = unicodedata.category(char)
        if char.isalnum() or category.startswith("M"):
            current.append(char)
        elif current:
            tokens.append("".join(current))
            current = []
    if current:
        tokens.append("".join(current))
    return [token for token in tokens if _has_letter(token)]


def _has_letter(text: str) -> bool:
    return any(char.isalpha() for char in text)


def _indic_letter_ratio(text: str) -> float:
    letters = [char for char in text if char.isalpha()]
    if not letters:
        return 0.0
    indic_letters = [
        char
        for char in letters
        if "\u0900" <= char <= "\u097f"
        or "\u0980" <= char <= "\u09ff"
        or "\u0a00" <= char <= "\u0a7f"
        or "\u0a80" <= char <= "\u0aff"
        or "\u0b00" <= char <= "\u0b7f"
        or "\u0b80" <= char <= "\u0bff"
        or "\u0c00" <= char <= "\u0c7f"
        or "\u0c80" <= char <= "\u0cff"
        or "\u0d00" <= char <= "\u0d7f"
    ]
    return len(indic_letters) / len(letters)


def _same_column(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> bool:
    center_a = (a[0] + a[2]) / 2
    center_b = (b[0] + b[2]) / 2
    return abs(a[0] - b[0]) < 52 or _horizontal_overlap_ratio(a, b) > 0.45 or abs(center_a - center_b) < 72


def _horizontal_overlap_ratio(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    overlap = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    narrower = max(1.0, min(a[2] - a[0], b[2] - b[0]))
    return overlap / narrower


def _center_inside(
    inner: tuple[float, float, float, float],
    outer: tuple[float, float, float, float],
) -> bool:
    center_x = (inner[0] + inner[2]) / 2
    center_y = (inner[1] + inner[3]) / 2
    return outer[0] <= center_x <= outer[2] and outer[1] <= center_y <= outer[3]


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

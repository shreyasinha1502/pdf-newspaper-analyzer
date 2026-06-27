from backend.services.image_detector import _dedupe_rects


def test_dedupes_overlapping_photo_rectangles():
    rects = [
        (10, 10, 210, 160),
        (20, 20, 205, 155),
        (300, 40, 460, 200),
    ]

    assert len(_dedupe_rects(rects)) == 2

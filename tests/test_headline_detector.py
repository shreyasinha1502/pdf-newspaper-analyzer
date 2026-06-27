from dataclasses import dataclass

from backend.services.headline_detector import detect_headlines


@dataclass(frozen=True)
class FakeRect:
    width: float = 600
    height: float = 800


@dataclass(frozen=True)
class FakeLine:
    text: str
    bbox: tuple[float, float, float, float]
    font_size: float
    bold: bool = False

    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]


def test_detects_large_bold_headlines_and_ignores_body_copy():
    lines = [
        FakeLine("Markets rally after policy shift", (40, 80, 430, 110), 22, True),
        FakeLine("This is normal paragraph text that should not count.", (42, 130, 540, 144), 10, False),
        FakeLine("Weather", (42, 164, 102, 178), 13, True),
        FakeLine("City council approves riverfront plan", (42, 240, 460, 266), 19, True),
    ]

    result = detect_headlines(lines, FakeRect())

    assert result.count == 2
    assert result.confidence > 0.6


def test_merges_headline_split_over_two_lines():
    lines = [
        FakeLine("New transit plan reshapes", (40, 90, 360, 116), 22, True),
        FakeLine("downtown commute", (41, 121, 300, 147), 21, True),
        FakeLine("Paragraph text remains ordinary.", (40, 170, 520, 184), 10, False),
    ]

    result = detect_headlines(lines, FakeRect())

    assert result.count == 1


def test_rejects_common_newspaper_false_positives():
    lines = [
        FakeLine("State budget clears final hurdle", (40, 90, 400, 118), 22, True),
        FakeLine("By Staff Reporter", (42, 124, 190, 140), 14, True),
        FakeLine("Photo by City Desk", (42, 155, 210, 170), 13, True),
        FakeLine("Monday, January 12, 2026", (42, 180, 260, 198), 15, True),
        FakeLine("This is a long body sentence, written in normal style, and it should not count.", (42, 230, 560, 246), 12, False),
        FakeLine("ADVERTISEMENT", (42, 280, 210, 300), 18, True),
        FakeLine("SPORTS", (42, 320, 120, 340), 20, True),
    ]

    result = detect_headlines(lines, FakeRect())

    assert result.count == 1


def test_detects_hindi_devanagari_headlines():
    lines = [
        FakeLine("अग्नि सुरक्षा की परीक्षा में पिछड़े जिले के कई अस्पताल", (25, 60, 575, 102), 30, True),
        FakeLine("यह सामान्य खबर की सामग्री है जिसे हेडलाइन नहीं माना जाना चाहिए।", (32, 125, 560, 142), 11, False),
        FakeLine("सो रही महिला के जेवर और", (320, 180, 560, 208), 21, True),
        FakeLine("मोबाइल की चोरी, प्राथमिकी", (320, 212, 560, 240), 21, True),
        FakeLine("विशेष अभियान में 29", (40, 300, 210, 322), 17, True),
        FakeLine("आरोपित हुए गिरफ्तार", (40, 325, 215, 347), 17, True),
        FakeLine("हथियार व शराब बरामद", (40, 350, 225, 372), 17, True),
    ]

    result = detect_headlines(lines, FakeRect())

    assert result.count == 3


def test_rejects_caption_below_photo_region():
    lines = [
        FakeLine("Flood rescue teams reach old city", (40, 60, 420, 90), 22, True),
        FakeLine("Residents wait near a flooded street on Friday", (60, 342, 420, 358), 12, True),
    ]
    image_rects = [(50, 130, 430, 330)]

    result = detect_headlines(lines, FakeRect(), image_rects=image_rects)

    assert result.count == 1


def test_subheadline_is_merged_with_parent_headline():
    lines = [
        FakeLine("Court orders inquiry into land deal", (40, 80, 440, 110), 24, True),
        FakeLine("Officials asked to submit report within two weeks", (42, 118, 430, 140), 16, True),
        FakeLine("Another ward gets new drainage plan", (42, 230, 390, 256), 21, True),
    ]

    result = detect_headlines(lines, FakeRect())

    assert result.count == 2


def test_counts_compact_hindi_column_headlines():
    lines = [
        FakeLine("बाइक सवार घायल", (40, 70, 158, 90), 14.5, True),
        FakeLine("तीन वारंटी गिरफ्तार", (190, 70, 335, 91), 15, True),
        FakeLine("यह सामान्य खबर का लंबा वाक्य है जिसे हेडलाइन नहीं माना जाना चाहिए।", (40, 115, 360, 130), 11, False),
        FakeLine("फोटो: घटनास्थल पर मौजूद लोग", (45, 342, 235, 356), 12, True),
    ]
    image_rects = [(40, 180, 250, 330)]

    result = detect_headlines(lines, FakeRect(), image_rects=image_rects)

    assert result.count == 2


def test_counts_compact_english_column_headlines_without_counting_labels():
    lines = [
        FakeLine("Bridge work delayed", (42, 70, 182, 88), 13.4, True),
        FakeLine("Clinic opens today", (225, 70, 360, 88), 13.2, True),
        FakeLine("Sports", (410, 70, 460, 86), 14, True),
        FakeLine("By City Bureau", (42, 98, 145, 112), 12.5, True),
        FakeLine("Normal report text continues across this column.", (42, 128, 340, 142), 11, False),
    ]

    result = detect_headlines(lines, FakeRect())

    assert result.count == 2


def test_counts_two_word_and_nonbold_prominent_headlines():
    lines = [
        FakeLine("सड़क हादसा", (40, 70, 145, 90), 14.2, True),
        FakeLine("जल संकट", (190, 70, 285, 90), 14.1, True),
        FakeLine("Power outage", (330, 70, 435, 90), 14.5, True),
        FakeLine("School inspection", (40, 145, 210, 167), 16.2, False),
        FakeLine("यह सामान्य खबर का लंबा वाक्य है जिसे हेडलाइन नहीं माना जाना चाहिए।", (40, 190, 360, 205), 11, False),
    ]

    result = detect_headlines(lines, FakeRect())

    assert result.count == 4

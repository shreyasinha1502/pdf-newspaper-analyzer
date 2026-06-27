---
title: PDF Newspaper Analyzer
emoji: 📰
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
---

# PDF Newspaper Analyzer

FastAPI application that analyzes uploaded PDFs and returns page count, total headline count, total photo count, processing time, confidence scores, and page-level processing logs. It uses only free and open-source tooling.

## Features

- Digital PDF parsing with PyMuPDF text spans.
- Scanned PDF support with PaddleOCR first and Tesseract fallback.
- Headline detection using font size, boldness, line geometry, uppercase ratio, page position, and split-line merging.
- Photo detection from embedded PDF images and OpenCV contour analysis for scanned pages.
- Drag-and-drop web UI with progress and responsive result cards.
- Swagger docs at `/docs`, health endpoint at `/health`, CORS, file size limits, page limits, and temporary-file cleanup.

## Requirements

- Python 3.11
- Tesseract installed locally if you want the fallback OCR engine.
- Docker is recommended for deployment because it installs system OCR dependencies.

## Run Locally

```bash
pip install -r requirements.txt
uvicorn backend.main:app --reload
```

Open `http://127.0.0.1:8000`.

## API

`POST /analyze`

Multipart form field:

- `file`: PDF file

Example response:

```json
{
  "pages": 8,
  "headlines": 43,
  "photos": 9,
  "headline_confidence": 0.81,
  "photo_confidence": 0.88,
  "logs": [
    {"page": 1, "mode": "digital", "headlines": 6, "photos": 2}
  ],
  "processing_time": "1.40 sec"
}
```

## Configuration

Environment variables use the `PDF_ANALYZER_` prefix.

- `PDF_ANALYZER_MAX_FILE_SIZE_MB`: default `100`
- `PDF_ANALYZER_MAX_PAGES`: default `300`
- `PDF_ANALYZER_MAX_WORKERS`: default `4`
- `PDF_ANALYZER_OCR_DPI`: default `180`

## Docker

```bash
docker build -t pdf-newspaper-analyzer .
docker run -p 8000:8000 pdf-newspaper-analyzer
```

## Deploy to Render

1. Push this repository to GitHub.
2. Create a new Render Blueprint from the repository.
3. Render will use `render.yaml` and the Dockerfile.
4. Open the service URL and test `/health`.

The free Render plan has limited memory and cold starts. For very large scanned PDFs, set `PDF_ANALYZER_MAX_WORKERS=1` or `2`.

## Deploy to HuggingFace Spaces

1. Create a new Docker Space.
2. Push this repository to the Space.
3. Keep the included `Dockerfile`.
4. The app starts with Uvicorn on port `8000`.

## Testing

```bash
pytest
```

The included tests cover the headline heuristics and image rectangle de-duplication. Add fixture PDFs for end-to-end regression tests when you have representative newspaper samples.

## Notes on Accuracy

Newspaper layouts vary heavily. The detector is intentionally conservative: it rejects dates, page numbers, small labels, weather/classified/advertisement markers, tiny symbols, separators, and very small image regions. Confidence scores are heuristic quality indicators, not statistical guarantees.

import tempfile
import time
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from backend.services.analyzer import PDFAnalyzer
from backend.utils.config import settings
from backend.utils.errors import AnalysisError

router = APIRouter(tags=["analysis"])


@router.post("/analyze")
async def analyze_pdf(file: UploadFile = File(...)) -> dict:
    started = time.perf_counter()

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF uploads are supported.",
        )

    content_type = (file.content_type or "").lower()
    if content_type and content_type not in {"application/pdf", "application/octet-stream"}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid upload content type.",
        )

    with tempfile.TemporaryDirectory(prefix="pdf-analyzer-") as temp_dir:
        temp_path = Path(temp_dir) / "upload.pdf"
        bytes_written = 0

        try:
            with temp_path.open("wb") as output:
                while chunk := await file.read(settings.upload_chunk_size):
                    bytes_written += len(chunk)
                    if bytes_written > settings.max_file_size_bytes:
                        raise HTTPException(
                            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                            detail=f"PDF exceeds the {settings.max_file_size_mb} MB upload limit.",
                        )
                    output.write(chunk)

            if bytes_written == 0:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Uploaded PDF is empty.",
                )

            result = PDFAnalyzer().analyze(temp_path)
            result["processing_time"] = f"{time.perf_counter() - started:.2f} sec"
            return result
        except HTTPException:
            raise
        except AnalysisError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="PDF analysis failed.",
            ) from exc
        finally:
            await file.close()

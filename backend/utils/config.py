from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PDF_ANALYZER_", env_file=".env", extra="ignore")

    max_file_size_mb: int = 100
    max_pages: int = 300
    max_workers: int = 4
    upload_chunk_size: int = 1024 * 1024
    ocr_dpi: int = 180
    ocr_languages: str = "eng+hin"
    paddleocr_language: str = "en"
    cors_origins: list[str] = ["*"]

    @property
    def max_file_size_bytes(self) -> int:
        return self.max_file_size_mb * 1024 * 1024


settings = Settings()

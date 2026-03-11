from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import io

# Pillow для чтения метаданных изображения
try:
    from PIL import Image
except ImportError:
    Image = None

app = FastAPI(
    title="Picture Utilities API",
    version="1.0.0",
    description="Provides image upload and object analysis for OpenWebUI tools.",
)

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Допустимые MIME-типы изображений
ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/png",
    "image/gif",
    "image/webp",
    "image/bmp",
}


# -------------------------------
# Pydantic models
# -------------------------------
class ImageInfo(BaseModel):
    filename: str = Field(..., description="Original filename")
    size_bytes: int = Field(..., description="File size in bytes")
    width: int = Field(..., description="Image width in pixels")
    height: int = Field(..., description="Image height in pixels")
    format: str = Field(..., description="Image format (e.g. JPEG, PNG)")
    content_type: str = Field(..., description="MIME type")


class ObjectAnalysisItem(BaseModel):
    label: str = Field(..., description="Object label or class")
    confidence: float | None = Field(None, description="Confidence score 0–1 if available")
    bbox: list[float] | None = Field(None, description="[x, y, w, h] normalized or pixels")


class AnalyzeImageResponse(BaseModel):
    success: bool = Field(True, description="Whether analysis succeeded")
    image: ImageInfo = Field(..., description="Image metadata")
    analysis: dict = Field(
        default_factory=dict,
        description="Analysis result (objects, description, etc.)",
    )
    message: str | None = Field(None, description="Optional message")


def get_image_info(data: bytes, filename: str, content_type: str) -> ImageInfo:
    """Извлекает метаданные изображения через Pillow."""
    if Image is None:
        raise HTTPException(
            status_code=500,
            detail="Pillow not installed; cannot read image metadata.",
        )
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid or corrupted image: {e}")
    return ImageInfo(
        filename=filename,
        size_bytes=len(data),
        width=img.width,
        height=img.height,
        format=img.format or "unknown",
        content_type=content_type,
    )


# -------------------------------
# Routes
# -------------------------------
@app.get("/")
def root():
    return {
        "service": "Picture Utilities API",
        "version": "1.0.0",
        "docs": "/docs",
        "openapi": "/openapi.json",
    }


@app.post(
    "/upload_photo_for_analysis",
    response_model=AnalyzeImageResponse,
    summary="Upload photo for object analysis",
    description="Accepts an image file and returns image metadata and analysis (e.g. dimensions, format). "
    "Extend this service or connect to a vision backend for full object detection.",
)
async def upload_photo_for_analysis(
    file: UploadFile = File(..., description="Image file (JPEG, PNG, GIF, WebP, BMP)"),
):
    content_type = file.content_type or ""
    if content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid content type: {content_type}. Allowed: {', '.join(sorted(ALLOWED_CONTENT_TYPES))}",
        )
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")
    filename = file.filename or "image"
    image_info = get_image_info(data, filename, content_type)
    # Базовый «анализ»: только метаданные; список объектов можно заполнять
    # из внешнего сервиса (например HereYouAre) или локальной модели.
    analysis = {
        "summary": f"Image received: {image_info.width}x{image_info.height} {image_info.format}.",
        "objects": [],
    }
    return AnalyzeImageResponse(
        success=True,
        image=image_info,
        analysis=analysis,
        message="For detailed object detection, connect this tool to a vision service (e.g. HereYouAre).",
    )

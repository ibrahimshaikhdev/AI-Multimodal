from __future__ import annotations

from io import BytesIO

from PIL import Image, ImageFilter, ImageOps, UnidentifiedImageError


def preprocess_image_for_ocr(image_bytes: bytes, target_width: int = 1800) -> bytes:
    if not isinstance(image_bytes, (bytes, bytearray)) or not image_bytes:
        raise ValueError("Image content is required")

    try:
        with Image.open(BytesIO(image_bytes)) as image:
            rgb_image = image.convert("RGB")

            width, height = rgb_image.size
            if width < target_width:
                scale = target_width / max(1, width)
                rgb_image = rgb_image.resize(
                    (max(1, int(width * scale)), max(1, int(height * scale))),
                    Image.Resampling.LANCZOS,
                )

            grayscale = ImageOps.grayscale(rgb_image)
            grayscale = ImageOps.autocontrast(grayscale)
            grayscale = grayscale.filter(ImageFilter.MedianFilter(size=3))
            grayscale = grayscale.point(lambda pixel: 255 if pixel > 180 else 0)

            buffer = BytesIO()
            grayscale.save(buffer, format="PNG")
            return buffer.getvalue()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError("Invalid image content") from exc


__all__ = ["preprocess_image_for_ocr"]

from io import BytesIO

from PIL import Image

from backend.services.ocr_preprocessing import preprocess_image_for_ocr


def test_preprocess_image_for_ocr_returns_processed_bytes():
    image = Image.new("RGB", (200, 80), color="white")
    draw = image.load()
    for x in range(10, 190):
        for y in range(20, 60):
            if x % 11 == 0 or y % 13 == 0:
                draw[x, y] = (0, 0, 0)

    buffer = BytesIO()
    image.save(buffer, format="PNG")

    processed = preprocess_image_for_ocr(buffer.getvalue())

    assert processed
    assert isinstance(processed, bytes)

    restored = Image.open(BytesIO(processed))
    assert restored.size[0] > 0
    assert restored.size[1] > 0


def test_preprocess_image_for_ocr_rejects_invalid_bytes():
    try:
        preprocess_image_for_ocr(b"not an image")
        assert False, "Expected ValueError for invalid image bytes"
    except ValueError:
        pass

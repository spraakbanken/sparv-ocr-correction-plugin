import pytest
from sbx_ocr_correction_viklofg_sweocr import (
    OcrCorrector,
)


@pytest.fixture(scope="session")
def ocr_corrector() -> OcrCorrector:
    return OcrCorrector.default()

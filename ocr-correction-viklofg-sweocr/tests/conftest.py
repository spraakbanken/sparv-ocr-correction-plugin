import pytest
from ocr_correction import (
    DEFAULT_MODEL_NAME,
    DEFAULT_TOKENIZER_NAME,
    OcrCorrector,
)
from transformers import (  # type: ignore [import-untyped]
    AutoTokenizer,
    T5ForConditionalGeneration,
)


@pytest.fixture(scope="session")
def ocr_corrector() -> OcrCorrector:
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_TOKENIZER_NAME)
    model = T5ForConditionalGeneration.from_pretrained(DEFAULT_MODEL_NAME)
    return OcrCorrector(model=model, tokenizer=tokenizer)

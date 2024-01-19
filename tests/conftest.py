import pytest

from transformers import T5ForConditionalGeneration, AutoTokenizer
from sparv_ocr_suggestion import (
    DEFAULT_TOKENIZER_NAME,
    OcrSuggestor,
    DEFAULT_MODEL_NAME,
)


@pytest.fixture(scope="session")
def ocr_suggestor() -> OcrSuggestor:
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_TOKENIZER_NAME)
    model = T5ForConditionalGeneration.from_pretrained(DEFAULT_MODEL_NAME)
    return OcrSuggestor(model=model, tokenizer=tokenizer)

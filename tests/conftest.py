import pytest
from ocr_correction import (
    DEFAULT_MODEL_NAME,
    DEFAULT_TOKENIZER_NAME,
    OcrSuggestor,
)
from transformers import (  # type: ignore [import-untyped]
    AutoTokenizer,
    T5ForConditionalGeneration,
)


@pytest.fixture(scope="session")
def ocr_suggestor() -> OcrSuggestor:
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_TOKENIZER_NAME)
    model = T5ForConditionalGeneration.from_pretrained(DEFAULT_MODEL_NAME)
    return OcrSuggestor(model=model, tokenizer=tokenizer)

from typing import List, Optional

from sparv import api as sparv_api  # type: ignore [import-untyped]
from transformers import (  # type: ignore [import-untyped]
    AutoTokenizer,
    T5ForConditionalGeneration,
    pipeline,
)


def bytes_length(s: str) -> int:
    return len(s.encode("utf-8"))


def zip_and_diff(orig: List[str], sugg: List[str]) -> List[Optional[str]]:
    return [sw if sw != ow else None for (ow, sw) in zip(orig, sugg)]


TOK_SEP = " "
logger = sparv_api.get_logger(__name__)
TOKENIZER_REVISION = "68377bdc18a2ffec8a0533fef03b1c513a4dd49d"
TOKENIZER_NAME = "google/byt5-small"
MODEL_REVISION = "84b138048992271be7617ccb11056bbcb9b72262"
MODEL_NAME = "viklofg/swedish-ocr-correction"


class OcrCorrector:
    TEXT_LIMIT: int = 127

    def __init__(self, *, tokenizer, model) -> None:
        self.tokenizer = tokenizer
        self.model = model
        self.pipeline = pipeline(
            "text2text-generation", model=model, tokenizer=tokenizer
        )

    @classmethod
    def default(cls) -> "OcrCorrector":
        tokenizer = AutoTokenizer.from_pretrained(
            TOKENIZER_NAME, revision=TOKENIZER_REVISION
        )
        model = T5ForConditionalGeneration.from_pretrained(
            MODEL_NAME, revision=MODEL_REVISION
        )
        return cls(model=model, tokenizer=tokenizer)

    def calculate_corrections(self, text: List[str]) -> List[Optional[str]]:
        logger.debug("Analyzing '%s'", text)
        parts = []
        curr_part: List[str] = []
        curr_len = 0
        ocr_corrections: List[str] = []
        for word in text:
            len_word = bytes_length(word)
            if (curr_len + len_word + 1) > self.TEXT_LIMIT:
                parts.append(TOK_SEP.join(curr_part))
                curr_part, curr_len = [word], len_word
            else:
                curr_part.append(word)
                curr_len = len_word if curr_len == 0 else curr_len + len_word + 1
        if len(curr_part) > 0:
            parts.append(TOK_SEP.join(curr_part))
        for part in parts:
            suggested_text = self.pipeline(part)[0]["generated_text"]
            suggested_text = suggested_text.replace(",", " ,")
            suggested_text = suggested_text.replace(".", " .")
            ocr_corrections = ocr_corrections + suggested_text.split(TOK_SEP)

        if len(text) == len(ocr_corrections) + 1 and text[-1] != ocr_corrections[-1]:
            ocr_corrections.append(text[-1])
        return zip_and_diff(text, ocr_corrections)

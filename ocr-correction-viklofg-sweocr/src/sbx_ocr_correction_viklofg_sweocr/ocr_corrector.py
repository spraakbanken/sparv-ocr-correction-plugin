from typing import List, Optional

from sparv.api import get_logger
from transformers import AutoTokenizer, T5ForConditionalGeneration, pipeline

MODEL_NAME = "viklofg/swedish-ocr-correction"
MODEL_REVISION = "84b138048992271be7617ccb11056bbcb9b72262"
TOKENIZER_NAME = "google/byt5-small"
TOKENIZER_REVISION = "68377bdc18a2ffec8a0533fef03b1c513a4dd49d"
TOK_SEP = " "
logger = get_logger(__name__)


def zip_and_diff(orig: List[str], sugg: List[str]) -> List[Optional[str]]:
    return [sw if sw != ow else None for (ow, sw) in zip(orig, sugg, strict=True)]


def bytes_length(s: str) -> int:
    return len(s.encode("utf-8"))


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
            print(f"{word=}")
            len_word = bytes_length(word)
            if (curr_len + len_word + 1) > self.TEXT_LIMIT:
                parts.append(TOK_SEP.join(curr_part))
                curr_part, curr_len = [word], len_word
            else:
                curr_part.append(word)
                curr_len = len_word if curr_len == 0 else curr_len + len_word + 1
            print(f"{curr_part=} {curr_len=}")
        if len(curr_part) > 0:
            parts.append(TOK_SEP.join(curr_part))
        for part in parts:
            print(f"{part=}")
            suggested_text = self.pipeline(part)[0]["generated_text"]
            suggested_text = suggested_text.replace(",", " ,")
            suggested_text = suggested_text.replace(".", " .")
            print(f"{suggested_text=}")
            ocr_corrections += suggested_text.split(TOK_SEP)

        if len(text) == len(ocr_corrections) + 1 and text[-1] != ocr_corrections[-1]:
            ocr_corrections.append(text[-1])
        return zip_and_diff(text, ocr_corrections)

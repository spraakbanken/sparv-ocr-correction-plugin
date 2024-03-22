from typing import Optional

from sparv.api import (  # type: ignore [import-untyped]
    Annotation,
    Output,
    annotator,
    get_logger,
)
from transformers import (  # type: ignore [import-untyped]
    AutoTokenizer,
    T5ForConditionalGeneration,
    pipeline,
)

__description__ = "Calculating word neighbours by mask a word in a BERT model."

DEFAULT_MODEL_NAME = "viklofg/swedish-ocr-correction"
DEFAULT_TOKENIZER_NAME = "google/byt5-small"


__version__ = "0.1.0"

logger = get_logger(__name__)

TOK_SEP = " "


@annotator(
    "Word neighbour tagging with a masked Bert model",
)
def annotate_ocr_correction(
    out_ocr_correction: Output = Output(
        "<token>:ocr_correction_viklofg_sweocr.ocr-correction--viklofg-swedish-ocr",
        cls="ocr_correction_viklofg_sweocr",
        description="Neighbours from masked BERT (format: '|<word>:<score>|...|)",
    ),
    word: Annotation = Annotation("<token:word>"),
    sentence: Annotation = Annotation("<sentence>"),
) -> None:
    tokenizer_name = DEFAULT_TOKENIZER_NAME
    model_name = DEFAULT_MODEL_NAME
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    ocr_corrector = OcrCorrector(model=model, tokenizer=tokenizer)

    sentences, _orphans = sentence.get_children(word)
    token_word = list(word.read())
    out_ocr_correction_annotation = word.create_empty_attribute()

    logger.progress(total=len(sentences))  # type: ignore
    for sent in sentences:
        logger.progress()  # type: ignore
        sent_to_tag = [token_word[token_index] for token_index in sent]

        ocr_corrections = ocr_corrector.calculate_corrections(sent_to_tag)
        out_ocr_correction_annotation[:] = ocr_corrections

    logger.info("writing annotations")
    out_ocr_correction.write(out_ocr_correction_annotation)


class OcrCorrector:
    TEXT_LIMIT: int = 127

    def __init__(self, *, tokenizer, model) -> None:
        self.tokenizer = tokenizer
        self.model = model
        self.pipeline = pipeline(
            "text2text-generation", model=model, tokenizer=tokenizer
        )

    def calculate_corrections(self, text: list[str]) -> list[Optional[str]]:
        logger.debug("Analyzing '%s'", text)
        parts = []
        curr_part: list[str] = []
        curr_len = 0
        ocr_corrections: list[str] = []
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


def zip_and_diff(orig: list[str], sugg: list[str]) -> list[Optional[str]]:
    return [sw if sw != ow else None for (ow, sw) in zip(orig, sugg)]


def bytes_length(s: str) -> int:
    return len(s.encode("utf-8"))
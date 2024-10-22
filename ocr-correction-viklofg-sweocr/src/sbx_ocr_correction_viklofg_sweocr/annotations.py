"""Annotators for OCR correction."""

from sparv import api as sparv_api  # type: ignore [import-untyped]
from sparv.api import Annotation, Output, annotator  # type: ignore [import-untyped]

from sbx_ocr_correction_viklofg_sweocr.constants import PROJECT_NAME
from sbx_ocr_correction_viklofg_sweocr.ocr_corrector import OcrCorrector

logger = sparv_api.get_logger(__name__)


@annotator("OCR corrections as annotations", language=["swe"])
def annotate_ocr_correction(
    out_ocr_correction: Output = Output(
        f"{PROJECT_NAME}.ocr_correction",
        description="OCR Corrections spans from viklfog/swedish-ocr",
    ),
    out_ocr_correction_corr: Output = Output(
        f"{PROJECT_NAME}.ocr_correction:{PROJECT_NAME}.ocr-correction--viklofg-sweocr",
        # cls="sbx_ocr_correction_viklofg_sweocr",
        description="OCR Corrections from viklfog/swedish-ocr",
    ),
    word: Annotation = Annotation("<token:word>"),
    sentence: Annotation = Annotation("<sentence>"),
    token: Annotation = Annotation("<token>"),
) -> None:
    """Annotate tokens with OCR correction."""
    ocr_corrector = OcrCorrector.default()

    sentences, _orphans = sentence.get_children(word)
    token_word = list(word.read())
    token_spans = list(token.read_spans())

    out_ocr_spans = []
    out_ocr_correction_annotation = []

    logger.progress(total=len(sentences))  # type: ignore
    for sent in sentences:
        logger.progress()  # type: ignore
        sent_to_tag = [token_word[token_index] for token_index in sent]

        base_token = token_spans[sent[0]][0]
        logger.debug("base_token = %d", base_token)
        print(f"base_token = {base_token}")
        ocr_corrections = ocr_corrector.calculate_corrections(sent_to_tag)
        for span, ocr_correction in ocr_corrections:
            # new_span = (span[0] + base_token, span[1] + base_token)
            print(f"{token_spans=}")
            new_span = (token_spans[span[0]][0], token_spans[span[1]][1])
            logger.debug("new_span=%s, ocr_correction=%s", new_span, ocr_correction)
            out_ocr_spans.append(new_span)
            out_ocr_correction_annotation.append(ocr_correction)

    logger.info("writing annotations")
    logger.debug("spans=%s", out_ocr_spans)
    out_ocr_correction.write(out_ocr_spans)
    logger.debug("annotations=%s", out_ocr_correction_annotation)
    out_ocr_correction_corr.write(out_ocr_correction_annotation)

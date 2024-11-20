"""Annotatinos for Sparv."""

from typing import Optional

from sparv import api as sparv_api  # type: ignore [import-untyped]
from sparv.api import Annotation, Output, annotator  # type: ignore [import-untyped]

from sbx_ocr_correction_viklofg_sweocr.ocr_corrector import OcrCorrector

logger = sparv_api.get_logger(__name__)


@annotator("OCR corrections as annotations", language=["swe"])
def annotate_ocr_correction(
    out_ocr: Output = Output("sbx_ocr_correction_viklofg_sweocr.sbx-ocr-correction", cls="ocr_correction"),
    out_ocr_corr: Output = Output(
        "sbx_ocr_correction_viklofg_sweocr.sbx-ocr-correction:sbx_ocr_correction_viklofg_sweocr.ocr-correction--viklofg-sweocr",
        cls="ocr_correction:correction",
    ),
    word: Annotation = Annotation("<token:word>"),
    sentence: Annotation = Annotation("<sentence>"),
    token: Annotation = Annotation("<token>"),
) -> None:
    """Sparv annotator to compute OCR corrections as annotations."""
    ocr_corrector = OcrCorrector.default()

    sentences, _orphans = sentence.get_children(word)
    token_word = list(word.read())

    ocr_corrections = []

    logger.progress(total=len(sentences))  # type: ignore
    for sent_idx in sentences:
        logger.progress()  # type: ignore
        sent = [token_word[token_index] for token_index in sent_idx]

        ocr_corrections.append(ocr_corrector.calculate_corrections(sent))

    parse_ocr_corrections(sentences, token, ocr_corrections, out_ocr, out_ocr_corr)


def parse_ocr_corrections(
    sentences: list,
    token: Annotation,
    ocr_corrections: list[list[tuple[tuple[int, int], Optional[str]]]],
    out_ocr: Output,
    out_ocr_corr: Output,
) -> None:
    """Parse OCR corrections and write output."""
    ocr_spans = []
    ocr_corr_ann = []

    token_spans = list(token.read_spans())
    for sent, corr_sent in zip(sentences, ocr_corrections):
        i = 0
        for span, corr_opt in corr_sent:
            start_pos = token_spans[sent[i]][0]

            i += span[1] - span[0]

            end_pos = token_spans[sent[i - 1]][1]
            logger.debug(
                "(%d, %d): '%s'",
                start_pos,
                end_pos,
                "" if corr_opt is None else corr_opt,
            )
            if corr_opt is not None:
                ocr_spans.append((start_pos, end_pos))
                ocr_corr_ann.append(corr_opt)

    logger.info("writing annotations")
    out_ocr.write(ocr_spans)
    out_ocr_corr.write(ocr_corr_ann)

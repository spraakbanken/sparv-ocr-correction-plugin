from sparv import api as sparv_api  # type: ignore [import-untyped]
from sparv.api import Annotation, Output, annotator  # type: ignore [import-untyped]

from sbx_ocr_correction_viklofg_sweocr.ocr_corrector import OcrCorrector

logger = sparv_api.get_logger(__name__)


@annotator("OCR corrections as annotations", language=["swe"])
def annotate_ocr_correction(
    out_ocr_correction: Output = Output(
        "<token>:sbx_ocr_correction_viklofg_sweocr.ocr-correction--viklofg-sweocr",
        cls="sbx_ocr_correction_viklofg_sweocr",
        description="OCR Corrections from viklfog/swedish-ocr (format: '|<word>:<score>|...|)",  # noqa: E501
    ),
    word: Annotation = Annotation("<token:word>"),
    sentence: Annotation = Annotation("<sentence>"),
) -> None:
    ocr_corrector = OcrCorrector.default()

    sentences, _orphans = sentence.get_children(word)
    token_word = list(word.read())
    out_ocr_correction_annotation = word.create_empty_attribute()

    logger.progress(total=len(sentences))  # type: ignore
    for sent in sentences:
        logger.progress()  # type: ignore
        sent_to_tag = [token_word[token_index] for token_index in sent]

        ocr_corrections = ocr_corrector.calculate_corrections(sent_to_tag)
        for i, ocr_correction in enumerate(ocr_corrections, start=sent[0]):
            out_ocr_correction_annotation[i] = ocr_correction

    logger.info("writing annotations")
    out_ocr_correction.write(out_ocr_correction_annotation)

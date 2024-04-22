from sparv.api import Annotation, Output, annotator, get_logger

from sbx_ocr_correction_viklofg_sweocr.ocr_corrector import OcrCorrector

logger = get_logger(__name__)


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
    logger.debug(
        "len(out_ocr_correction_annotation) = %d", len(out_ocr_correction_annotation)
    )

    logger.progress(total=len(sentences))  # type: ignore
    for sent in sentences:
        logger.progress()  # type: ignore
        sent_to_tag = [token_word[token_index] for token_index in sent]
        logger.debug("sent = %s", sent)
        logger.debug("len(sent) = %d", len(sent))

        ocr_corrections = ocr_corrector.calculate_corrections(sent_to_tag)
        logger.debug("len(ocr_corrections) = %d", len(ocr_corrections))
        logger.debug("ocr_corrections = %s", ocr_corrections)
        for i, ocr_correction in enumerate(ocr_corrections, start=sent[0]):
            out_ocr_correction_annotation[i] = ocr_correction
        logger.debug(
            "len(out_ocr_correction_annotation) = %d",
            len(out_ocr_correction_annotation),
        )
        logger.debug(
            "out_ocr_correction_annotation = %s",
            out_ocr_correction_annotation[sent[0] : sent[-1]],
        )

    logger.info("writing annotations")
    out_ocr_correction.write(out_ocr_correction_annotation)

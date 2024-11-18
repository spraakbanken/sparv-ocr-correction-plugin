from sbx_ocr_correction_viklofg_sweocr.annotations import annotate_ocr_correction
from sparv_pipeline_testing import MemoryOutput, MockAnnotation


def test_annotate_ocr_correction(snapshot) -> None:  # noqa: ANN001
    output_ocr: MemoryOutput = MemoryOutput()
    output_ocr_corr: MemoryOutput = MemoryOutput()
    # "Jonath an saknades ."
    # "12345678901234567890"
    # "         1         2"
    word = MockAnnotation(name="<token:word>", values=["Jonath", "an", "saknades", "."])
    sentence = MockAnnotation(name="<sentence>", children={"<token:word>": [[0, 1, 2, 3]]})
    token = MockAnnotation(name="<token>", spans=[(0, 6), (7, 9), (10, 18), (19, 20)])

    annotate_ocr_correction(output_ocr, output_ocr_corr, word, sentence, token)

    assert output_ocr.values == snapshot
    assert output_ocr_corr.values == snapshot

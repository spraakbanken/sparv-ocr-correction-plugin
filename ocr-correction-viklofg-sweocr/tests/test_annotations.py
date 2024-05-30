from sbx_ocr_correction_viklofg_sweocr.annotations import annotate_ocr_correction
from sparv_pipeline_testing import MemoryOutput, MockAnnotation


def test_annotate_sentence_sentiment(snapshot) -> None:
    output: MemoryOutput = MemoryOutput()
    word = MockAnnotation(
        name="<token:word>",
        values=[
            "Den",
            "i",
            "HandelstidniDgens",
            "g&rdagsnnmmer",
            "omtalade",
            "hvalfisken",
            ",",
            "sorn",
            "fångats",
            "i",
            "Frölnndaviken",
            ".",
        ],
        children={"<token:word>": [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]},
    )
    sentence = MockAnnotation(
        name="<sentence>", children={"<token:word>": [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]}
    )

    annotate_ocr_correction(output, word=word, sentence=sentence)

    assert output.values == snapshot

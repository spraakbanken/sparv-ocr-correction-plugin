from sbx_ocr_correction_viklofg_sweocr.annotations import annotate_ocr_correction
from sparv_pipeline_testing import MemoryOutput, MockAnnotation


def test_annotate_ocr_correction(snapshot) -> None:
    output_spans: MemoryOutput = MemoryOutput()
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
    token = MockAnnotation(name="<token>", spans=[(0, 1), (1, 2), (2, 3), (3, 4), (5, 6)])

    annotate_ocr_correction(output_spans, output, word=word, sentence=sentence, token=token)

    assert output.values == snapshot


def test_correction_spans_2_tokens(snapshot) -> None:
    # TODO titta på original texten
    output_spans: MemoryOutput = MemoryOutput()
    output_annotations: MemoryOutput = MemoryOutput()
    word = MockAnnotation(
        name="<token:word>",
        values="Må de se ' n åt verlden skaffa , om de kunna , Bättre dar än våra , bättre verk än vårt !".split(
            " "
        ),
    )
    sentence = MockAnnotation(
        name="<sentence>", children={"<token:word>": [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]}
    )
    token = MockAnnotation(name="<token>", spans=[(0, 3)])

    annotate_ocr_correction(
        output_spans, output_annotations, word=word, sentence=sentence, token=token
    )

    assert output_spans.values == snapshot
    assert output_annotations.values == snapshot

from sparv_ocr_suggestion import OcrSuggestor


def test_short_text(ocr_suggestor: OcrSuggestor):
    text = "Den i HandelstidniDgens g&rdagsnnmmer omtalade hvalfisken , sorn fångats i Frölnndaviken ."
    actual = ocr_suggestor.calculate_suggestions(text)

    expected = [
        None,
        None,
        "Handelstidningens",
        "gårdagsnummer",
        None,
        None,
        None,
        "som",
        None,
        None,
        "Frölandsviken",
        None,
    ]
    assert actual == expected

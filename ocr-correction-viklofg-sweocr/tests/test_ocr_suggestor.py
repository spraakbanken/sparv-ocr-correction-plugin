from sbx_ocr_correction_viklofg_sweocr import OcrCorrector


def test_short_text(ocr_corrector: OcrCorrector):
    text = [
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
    ]
    actual = ocr_corrector.calculate_corrections(text)

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


def test_long_text(ocr_corrector: OcrCorrector):
    text1 = [
        "Förvaltningen",
        "af",
        "Biksgäldskontoret",
        ",",
        "dess",
        "medel",
        "och",
        "tillhörigheter",
        "är",
        "utaf",
        "Riksdagen",
        "uppdragen",
        "åt",
        "sju",
        "Fullmäktige",
        ",",
        "hvilka",
        "vid",
        "lagtima",
        "riksdag",
        "utses",
        "i",
        "den",
        "ordning",
        "71",
        "§",
        "Riksdagsordningcn",
        "stadgar",
        ".",
    ]
    # text2 = """Vid Fullmäktiges sammankomster föres ordet af den, som af Riksdagen
    # blifvit dertill utsedd; tillkommande Fullmäktige att sjelfva bland sig välja en
    # vice Ordförande att föra ordet, när hinder för Ordföranden inträffar."""
    # print(f"{len(text2)=}, {len(text2.encode())=}")
    actual = ocr_corrector.calculate_corrections(text1)
    # actual = ocr_corrector.calculate_corrections(text2)

    expected = [
        None,
        None,
        "Riksgäldskontoret",
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        "Riksdagsordningen",
        None,
        None,
    ]

    assert actual == expected

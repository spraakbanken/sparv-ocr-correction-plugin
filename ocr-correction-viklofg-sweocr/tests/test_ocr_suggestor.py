from sbx_ocr_correction_viklofg_sweocr.ocr_corrector import OcrCorrector


def test_short_text(ocr_corrector: OcrCorrector, snapshot) -> None:  # noqa: ANN001
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

    assert actual == snapshot


def test_long_text(ocr_corrector: OcrCorrector, snapshot) -> None:  # noqa: ANN001
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

    assert actual == snapshot


def test_issue_40(ocr_corrector: OcrCorrector, snapshot) -> None:  # noqa: ANN001
    example = [
        "Jonathan",
        "saknades",
        ",",
        "emedan",
        "han",
        ",",
        "med",
        "sin",
        "vapendragare",
        ",",
        "redan",
        "på",
        "annat",
        "håll",
        "sökt",
        "och",
        "anträffat",
        "fienden",
        ".",
    ]

    actual = ocr_corrector.calculate_corrections(example)

    assert actual == snapshot

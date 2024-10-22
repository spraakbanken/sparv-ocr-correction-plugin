from sbx_ocr_correction_viklofg_sweocr.ocr_corrector import OcrCorrector


def test_short_text(ocr_corrector: OcrCorrector, snapshot) -> None:
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


def test_long_text(ocr_corrector: OcrCorrector, snapshot) -> None:
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


def test_issue_40(ocr_corrector: OcrCorrector, snapshot) -> None:
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


def test_issue_44(ocr_corrector: OcrCorrector) -> None:
    example = [
        "Alla",
        "de",
        "andra",
        "voro",
        "till",
        "hands",
        ",",
        "stridbare",
        ",",
        "karske",
        "män",
        ",",
        "så",
        "länge",
        "kraften",
        "stod",
        "bi",
        ",",
        "och",
        "till",
        "dess",
        "äfven",
        "de",
        ",",
        "hvar",
        "efter",
        "annan",
        ",",
        "förr",
        "eller",
        "sednare",
        "föllo",
        "till",
        "jorden",
        ",",
        "och",
        "vapnen",
        "ur",
        "deras",
        "domnande",
        "händer",
        ".",
    ]

    actual = ocr_corrector.calculate_corrections(example)

    for i in range(len(actual) - 1):
        value_i = actual[i][0]
        value_i_1 = actual[i + 1][0]
        assert value_i <= value_i_1

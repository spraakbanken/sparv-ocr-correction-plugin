import pytest
from sbx_ocr_correction_viklofg_sweocr.ocr_corrector import OcrCorrector


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


@pytest.mark.parametrize(
    "text",
    [
        [
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
        ],
        [
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
        ],
    ],
)
def test_texts_issue_39(text: list[str], ocr_corrector: OcrCorrector, snapshot) -> None:
    """Test cases related to https://github.com/spraakbanken/sparv-sbx-ocr-correction/issues/39."""

    actual = ocr_corrector.calculate_corrections(text)

    assert len(actual) == len(text)
    assert actual == snapshot

import re
from typing import List, TypedDict

from sbx_ocr_correction_viklofg_sweocr.graph import utils


class Text(TypedDict):
    text: str


class Token(Text, TypedDict):
    id: str


class Span(TypedDict):
    begin: int
    end: int


def text(ts: List[Text]) -> str:
    """The text in some tokens

    >>> text(identify(tokenize('apa bepa cepa '), '#'))
    'apa bepa cepa '

    """
    return "".join(texts(ts))


def texts(ts: List[Text]) -> List[str]:
    """The texts in some tokens

    >>> texts(identify(tokenize('apa bepa cepa '), '#'))
    ['apa ', 'bepa ', 'cepa ']
    """
    return [t["text"] for t in ts]


def tokenize(s: str) -> List[str]:
    """Tokenizes text on whitespace, prefers to have trailing whitespace."""
    return list(
        map(
            utils.end_with_space,
            re.findall(r"\s*\S+\s*", s) or re.findall(r"^\s+$", s) or [],
        )
    )


def identify(toks: List[str], prefix: str) -> List[Token]:
    return [Token(text=text, id=f"{prefix}{i}") for i, text in enumerate(toks)]

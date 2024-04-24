import enum
from typing import Dict, List, Optional, TypedDict, TypeVar

from sbx_ocr_correction_viklofg_sweocr.graph import token, utils
from sbx_ocr_correction_viklofg_sweocr.graph.token import Token

A = TypeVar("A")


class Side(enum.StrEnum):
    source = "source"
    target = "target"


class SourceTarget(TypedDict):
    source: List[Token]
    target: List[Token]


class Edge(TypedDict):
    # a copy of the identifier used in the edges object of the graph
    id: str
    # these are ids to source and target tokens
    ids: List[str]
    # labels on this edge
    labels: List[str]
    # is this manually or automatically aligned
    manual: bool
    comment: Optional[str] = None


Edges = dict[str, Edge]


class Graph(SourceTarget):
    edges: Edges
    comment: Optional[str] = None


def edge(
    ids: List[str],
    labels: List[str],
    *,
    comment: Optional[str] = None,
    manual: bool = False,
) -> Edge:
    ids_sorted = sorted(ids)
    labels_nub = utils.uniq(labels)
    return Edge(
        id=f"e-{'-'.join(ids_sorted)}",
        ids=ids_sorted,
        labels=labels_nub,
        manual=manual,
        comment=comment,
    )


def edge_record(es: List[Edge]) -> Dict[str, Edge]:
    return {e["id"]: e for e in es}


def init(s: str, *, manual: bool = False) -> Graph:
    return init_from(token.tokenize(s), manual=manual)


def init_from(tokens: List[str], *, manual: bool = False) -> Graph:
    return {
        "source": token.identify(tokens, "s"),
        "target": token.identify(tokens, "t"),
        "edges": edge_record(
            (edge([f"s{i}", f"t{i}"], [], manual=manual) for i, _ in enumerate(tokens))
        ),
    }


def unaligned_set_side(g: Graph, side: Side, text: str) -> Graph:
    text0 = get_side_text(g, side)
    from_, to = utils.edit_range(text0, text)
    new_text = text[from_ : (len(text) - (len(text0) - to))]  # noqa: F841
    # return unaligned_modify(g, from_, to, new_text, side)


def get_side_text(g: Graph, side: Side) -> str:
    return token.text(g[side])

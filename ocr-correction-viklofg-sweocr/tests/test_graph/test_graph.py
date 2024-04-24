from sbx_ocr_correction_viklofg_sweocr.graph import graph


def test_graph_init() -> None:
    g = graph.init("w1 w2")
    source = [{"text": "w1 ", "id": "s0"}, {"text": "w2 ", "id": "s1"}]
    target = [{"text": "w1 ", "id": "t0"}, {"text": "w2 ", "id": "t1"}]
    edges = graph.edge_record(
        [graph.edge(["s0", "t0"], []), graph.edge(["s1", "t1"], [])]
    )

    assert g == {"source": source, "target": target, "edges": edges}


def test_graph_align() -> None:
    g0 = graph.init("a bc d")
    g = graph.unaligned_set_side(g0, "target", "ab c d")

    assert len(graph.align(g).edges) == 2

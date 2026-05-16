from graph_recsys.inference.recommend import recommend_top_k


def test_recommend_top_k_respects_limit() -> None:
    out = recommend_top_k("u1", ["i1", "i2", "i3"], k=2)
    assert out == ["i1", "i2"]

from graph_recsys.evaluation.metrics import accuracy, auc_roc, ndcg_at_k
from graph_recsys.models.fom import sample_masked_sequence
from graph_recsys.models.pog_proxy import should_stop_decode


def test_mask_sequence() -> None:
    context, target, pos = sample_masked_sequence(["i1", "i2", "i3"])
    assert target == "i2"
    assert pos == 1
    assert context == ["i1", "i3"]


def test_decode_stop_conditions() -> None:
    assert should_stop_decode("[END]", 2)
    assert should_stop_decode("i3", 8)
    assert not should_stop_decode("i3", 4)


def test_metrics_on_toy() -> None:
    assert accuracy([1, 1, 1], [1, 0, 1]) == 2 / 3
    auc = auc_roc([1, 0, 1, 0], [0.9, 0.1, 0.8, 0.2])
    assert 0.9 <= auc <= 1.0
    assert ndcg_at_k(1, 10) > ndcg_at_k(2, 10) > 0

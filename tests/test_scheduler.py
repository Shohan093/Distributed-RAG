import numpy as np

from driftdrag.schedule.scheduler import QualityDropPredictor, ShardCandidate, schedule_shards, total_cost


def test_quality_drop_predictor_fits_linear_signal():
    rng = np.random.default_rng(0)
    anchor_res = rng.uniform(0, 1, size=200)
    data_drift = rng.uniform(0, 1, size=200)
    drops = 0.5 * anchor_res + 0.3 * data_drift + rng.normal(0, 0.01, size=200)

    predictor = QualityDropPredictor().fit(anchor_res, data_drift, drops)
    preds = predictor.predict(anchor_res, data_drift)
    correlation = np.corrcoef(preds, drops)[0, 1]
    assert correlation > 0.95


def test_schedule_shards_prefers_best_ratio_under_tight_budget():
    candidates = [
        # cheap, big recovery -> should be picked first
        ShardCandidate("shard-0", predicted_drop_compensate=0.5, predicted_drop_partial=0.1, predicted_drop_full=0.0, cost_partial=1.0, cost_full=2.0),
        # expensive, small recovery -> should be skipped under a tight budget
        ShardCandidate("shard-1", predicted_drop_compensate=0.2, predicted_drop_partial=0.18, predicted_drop_full=0.15, cost_partial=10.0, cost_full=20.0),
    ]
    decisions = schedule_shards(candidates, budget=1.0)
    by_id = {d.shard_id: d for d in decisions}
    assert by_id["shard-0"].action == "partial"
    assert by_id["shard-1"].action == "compensate"
    assert total_cost(decisions) <= 1.0 + 1e-9


def test_schedule_shards_zero_budget_is_all_compensate():
    candidates = [
        ShardCandidate("shard-0", 0.5, 0.1, 0.0, 1.0, 2.0),
        ShardCandidate("shard-1", 0.3, 0.2, 0.1, 1.0, 2.0),
    ]
    decisions = schedule_shards(candidates, budget=0.0)
    assert all(d.action == "compensate" for d in decisions)


def test_schedule_shards_generous_budget_fully_reembeds_all():
    candidates = [
        ShardCandidate("shard-0", 0.5, 0.3, 0.0, 1.0, 2.0),
        ShardCandidate("shard-1", 0.4, 0.2, 0.0, 1.0, 2.0),
    ]
    decisions = schedule_shards(candidates, budget=100.0)
    assert all(d.action == "full" for d in decisions)

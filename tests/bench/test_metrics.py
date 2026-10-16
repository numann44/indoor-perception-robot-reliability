import numpy as np
import pytest
from robot_reliability_bench.metrics import box_iou, detection_ap, semantic_metrics, forecasting_metrics, tracking_metrics


def test_ap_correct_ranked_matching_and_duplicate_false_positive():
    targets = {"a": [{"bbox": [0, 0, 10, 10], "class_id": 1}], "b": [{"bbox": [0, 0, 10, 10], "class_id": 1}]}
    predictions = {"a": [{"bbox": [0, 0, 10, 10], "class_id": 1, "score": 0.9},
                          {"bbox": [0, 0, 10, 10], "class_id": 1, "score": 0.8}],
                   "b": [{"bbox": [0, 0, 10, 10], "class_id": 1, "score": 0.7}]}
    result = detection_ap(predictions, targets)
    assert result["AP"] < 1
    assert result["AP"] > 0.8
    assert detection_ap({"a": predictions["a"][:1], "b": predictions["b"]}, targets)["AP"] == pytest.approx(1)
    assert detection_ap({}, targets)["AP"] == 0


def test_ignore_suppresses_false_positive():
    target = {"a": [{"bbox": [0, 0, 10, 10], "class_id": 1},
                    {"bbox": [20, 20, 30, 30], "class_id": 1, "ignore": True}]}
    predictions = {"a": [{"bbox": [20, 20, 30, 30], "class_id": 1, "score": 1},
                          {"bbox": [0, 0, 10, 10], "class_id": 1, "score": 0.5}]}
    assert detection_ap(predictions, target)["AP"] == pytest.approx(1)
    assert detection_ap(predictions, {})["status"] == "unavailable"
    assert box_iou([], [target["a"][0]["bbox"]]).shape == (0, 1)


def test_confusion_ignores_unknown_and_counts_dropped_frame():
    target = {"a": np.array([[0, 1], [1, 255]]), "b": np.ones((2, 2), dtype=int)}
    pred = {"a": np.array([[0, 1], [0, 0]]), "b": None}
    result = semantic_metrics(pred, target, num_classes=2)
    assert result["confusion_matrix"] == [[1, 0], [1, 1]]
    assert result["missed_ground_truth_pixels"] == [0, 4]
    assert result["per_class_IoU"] == pytest.approx([0.5, 1 / 6])
    assert result["pixel_accuracy"] == pytest.approx(2 / 7)


def test_forecast_units_and_horizons():
    targets = np.zeros((2, 20, 2))
    predicted = targets.copy(); predicted[:, :, 0] = 3; predicted[:, :, 1] = 4
    result = forecasting_metrics(predicted, targets, unit="px", horizons=[0.5, 1, 2, 3])
    assert result["ADE"] == 5
    assert result["FDE"] == 5
    assert result["horizons"]["2"]["FDE"] == 5
    assert result["horizons"]["3"]["status"] == "unavailable"
    with pytest.raises(ValueError):
        forecasting_metrics(predicted, targets, unit="ambiguous")


def test_tracking_global_assignment_switches_and_full_string_suffixes():
    gt = {f"frame-{i}": [{"bbox": [0, 0, 10, 10], "track_id": "video_0001:1_2_3b", "class_id": 1},
                         {"bbox": [20, 0, 30, 10], "track_id": "video_0001:1_2_3a", "class_id": 1}] for i in range(2)}
    pred = {"frame-0": [{"bbox": [0, 0, 10, 10], "track_id": "det-a"},
                         {"bbox": [20, 0, 30, 10], "track_id": "det-b"}],
            "frame-1": [{"bbox": [0, 0, 10, 10], "track_id": "det-b"},
                         {"bbox": [20, 0, 30, 10], "track_id": "det-a"}]}
    result = tracking_metrics(pred, gt)
    assert result["IDF1"] == 0.5
    assert result["ID_switch"] == 2
    assert result["IDTP"] == 2
    assert result["IDFN"] == 2
    assert result["IDFP"] == 2
    assert {x["ground_truth_id"] for x in result["identity_assignment"]} == {"video_0001:1_2_3a", "video_0001:1_2_3b"}


def test_tracking_misses_false_positive_and_empty_inputs():
    gt = {"0": [{"bbox": [0, 0, 10, 10], "track_id": "GT-suffix-z"}],
          "1": [{"bbox": [0, 0, 10, 10], "track_id": "GT-suffix-z"}]}
    pred = {"0": [{"bbox": [0, 0, 10, 10], "track_id": "a"},
                   {"bbox": [20, 0, 30, 10], "track_id": "extra"}], "1": []}
    result = tracking_metrics(pred, gt)
    assert result["IDTP"] == 1
    assert result["IDFN"] == 1
    assert result["IDFP"] == 1
    assert result["IDF1"] == 0.5
    assert result["missed_detections"] == 1
    assert result["false_positive"] == 1
    assert tracking_metrics({}, gt)["IDF1"] == 0
    assert tracking_metrics(pred, {})["status"] == "unavailable"
    assert tracking_metrics(pred, {"0": [{"bbox": [0, 0, 10, 10]}]})["status"] == "unavailable"
    assert tracking_metrics({"0": [{"bbox": [0, 0, 10, 10]}]}, gt)["status"] == "unavailable"


def test_tracking_global_identity_is_not_greedy_frame_identity():
    # Framewise closest-box choices switch, but both predictions overlap GT at
    # every frame. Global identity assignment must preserve the full overlap.
    gt = {"0": [{"bbox": [0, 0, 10, 10], "track_id": "g"}],
          "1": [{"bbox": [0, 0, 10, 10], "track_id": "g"}]}
    pred = {"0": [{"bbox": [0, 0, 10, 10], "track_id": "a"}, {"bbox": [1, 0, 11, 10], "track_id": "b"}],
            "1": [{"bbox": [1, 0, 11, 10], "track_id": "a"}, {"bbox": [0, 0, 10, 10], "track_id": "b"}]}
    result = tracking_metrics(pred, gt)
    assert result["IDTP"] == 2
    assert result["IDF1"] == pytest.approx(2 / 3)
    assert result["ID_switch"] == 1


def test_duplicate_identity_in_one_frame_rejected():
    with pytest.raises(ValueError, match="Duplicate"):
        tracking_metrics({"0": [{"bbox": [0, 0, 10, 10], "track_id": "a"},
                                 {"bbox": [20, 0, 30, 10], "track_id": "a"}]},
                         {"0": [{"bbox": [0, 0, 10, 10], "track_id": "g"}]})


def test_tracking_crowd_ignore_suppresses_only_unmatched_predictions():
    gt = {"0": [{"bbox": [0, 0, 10, 10], "track_id": "g"},
                 {"bbox": [0, 0, 40, 40], "class_id": 1, "ignore": True}]}
    pred = {"0": [{"bbox": [0, 0, 10, 10], "track_id": "a"},
                   {"bbox": [20, 20, 25, 25], "track_id": "crowd-region"}]}
    result = tracking_metrics(pred, gt)
    assert result["IDF1"] == 1
    assert result["ignored_predictions"] == 1
    assert result["false_positive"] == 0

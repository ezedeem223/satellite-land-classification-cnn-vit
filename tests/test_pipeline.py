from satellite_land_classification.evaluate import compute_binary_metrics


def test_compute_binary_metrics_returns_expected_keys():
    metrics = compute_binary_metrics(
        y_true=[0, 1, 1, 0],
        y_pred=[0, 1, 0, 0],
        y_prob=[0.1, 0.9, 0.4, 0.2],
        label_names=["non-agri", "agri"],
    )
    assert set(metrics) == {
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "roc_auc",
        "confusion_matrix",
        "classification_report",
    }
    assert metrics["accuracy"] == 0.75


"""
Model insights and verified benchmark metrics API routes.
"""
from fastapi import APIRouter
from backend.app.schemas.insights import (
    MetricSplit,
    BenchmarksResponse,
    BranchArchitecture,
    ModelInfoResponse,
)

router = APIRouter(tags=["Model Insights"])


@router.get(
    "/benchmarks",
    response_model=BenchmarksResponse,
    summary="Retrieve verified validation and test performance benchmarks",
)
def get_benchmarks():
    """
    Return reproduced, strictly verified classification metrics.
    Metrics are clearly separated by split, evaluated checkpoint, and decision threshold.
    """
    return BenchmarksResponse(
        validation_metrics_at_041=MetricSplit(
            split_name="Validation Split",
            sample_count=4626,
            checkpoint="best_fusion_model.zip (Epoch 7)",
            threshold=0.41,
            roc_auc=89.12,
            balanced_accuracy=82.33,
            recall_sensitivity=79.21,
            specificity=85.45,
            precision=55.64,
            f1_score=65.36,
        ),
        validation_metrics_at_050=MetricSplit(
            split_name="Validation Split",
            sample_count=4626,
            checkpoint="best_fusion_model.zip (Epoch 7)",
            threshold=0.50,
            roc_auc=89.12,
            balanced_accuracy=82.39,
            recall_sensitivity=75.98,
            specificity=88.80,
            precision=60.98,
            f1_score=67.66,
        ),
        test_metrics_at_041=MetricSplit(
            split_name="Holdout Test Split",
            sample_count=5442,
            checkpoint="best_fusion_model.zip (Epoch 7)",
            threshold=0.41,
            roc_auc=88.44,
            balanced_accuracy=81.01,
            recall_sensitivity=76.51,
            specificity=85.51,
            precision=53.14,
            f1_score=62.72,
        ),
        test_metrics_at_050=MetricSplit(
            split_name="Holdout Test Split",
            sample_count=5442,
            checkpoint="best_fusion_model.zip (Epoch 7)",
            threshold=0.50,
            roc_auc=88.44,
            balanced_accuracy=80.54,
            recall_sensitivity=72.56,
            specificity=88.53,
            precision=57.59,
            f1_score=64.21,
        ),
        provenance_note=(
            "All figures were reproduced directly using best_fusion_model.zip checkpoint (Epoch 7) "
            "and verified feature arrays. Threshold 0.41 reflects the saved optimal decision cutoff."
        ),
    )


@router.get(
    "/model-info",
    response_model=ModelInfoResponse,
    summary="Retrieve model architecture specifications and parameter count",
)
def get_model_info():
    """
    Return comprehensive architectural details of the multi-branch fusion system.
    """
    return ModelInfoResponse(
        model_name="CardioInsight ECGFusionClassifier",
        total_parameters=2514819,
        trainable_parameters=2514819,
        checkpoint_epoch=7,
        branches=[
            BranchArchitecture(
                name="ResNet34-1D",
                role="Morphological Feature Extraction",
                output_dimension=512,
                details="34-layer 1D deep residual network producing 512-dimensional morphological features (7,223,681 parameters).",
            ),
            BranchArchitecture(
                name="BiGRU",
                role="Temporal Dynamics Extraction",
                output_dimension=256,
                details="3-layer Bidirectional Gated Recurrent Unit with single-head additive (Bahdanau-style) temporal attention producing 256-dimensional features (806,169 parameters).",
            ),
        ],
        fusion_mechanism=(
            "Adaptive Reliability Fusion: Projects both feature representations (ResNet 512-d, BiGRU 256-d) into a shared "
            "512-dimensional latent space. Confidence-weighted Softmax gating dynamically assigns per-sample scalar weights "
            "(alpha_resnet, alpha_bigru, sum=1.0) followed by 8-head cross-modal multi-head self-attention."
        ),
        classifier_head=(
            "4-stage Multilayer Perceptron: 512 -> 512 -> 256 -> 64 -> 1 with LayerNorm, GELU activations, and Dropout (0.30 / 0.20)."
        ),
        combined_system_parameters=10544669,
    )

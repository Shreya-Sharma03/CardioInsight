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
            roc_auc=88.92,
            balanced_accuracy=82.35,
            recall_sensitivity=77.94,
            specificity=86.76,
            precision=57.54,
            f1_score=66.21,
        ),
        validation_metrics_at_050=MetricSplit(
            split_name="Validation Split",
            sample_count=4626,
            checkpoint="best_fusion_model.zip (Epoch 7)",
            threshold=0.50,
            roc_auc=88.92,
            balanced_accuracy=80.12,
            recall_sensitivity=70.59,
            specificity=89.65,
            precision=62.30,
            f1_score=66.19,
        ),
        test_metrics_at_041=MetricSplit(
            split_name="Holdout Test Split",
            sample_count=5442,
            checkpoint="best_fusion_model.zip (Epoch 7)",
            threshold=0.41,
            roc_auc=87.54,
            balanced_accuracy=80.59,
            recall_sensitivity=75.14,
            specificity=86.04,
            precision=55.10,
            f1_score=63.58,
        ),
        test_metrics_at_050=MetricSplit(
            split_name="Holdout Test Split",
            sample_count=5442,
            checkpoint="best_fusion_model.zip (Epoch 7)",
            threshold=0.50,
            roc_auc=87.54,
            balanced_accuracy=78.42,
            recall_sensitivity=67.80,
            specificity=89.04,
            precision=59.42,
            f1_score=63.33,
        ),
        provenance_note=(
            "All figures were reproduced directly using best_fusion_model.zip checkpoint (Epoch 7) "
            "and verified feature arrays. Threshold 0.41 reflects the notebook's saved decision cutoff."
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
                details="34-layer 1D deep residual network with adaptive average pooling and batch normalization.",
            ),
            BranchArchitecture(
                name="BiGRU",
                role="Temporal Dynamics Extraction",
                output_dimension=256,
                details="3-layer Bidirectional Gated Recurrent Unit with skip-connections and temporal multi-head attention.",
            ),
        ],
        fusion_mechanism=(
            "Adaptive Reliability Fusion: Confidence-weighted Softmax gating dynamically assigns per-sample scalar "
            "weights (alpha_resnet, alpha_bigru) to scale feature embeddings, followed by 8-head cross-modal self-attention."
        ),
        classifier_head=(
            "Multilayer Perceptron: 512 -> 256 -> 64 -> 1 with LayerNorm, ReLU activations, and Dropout (0.3 / 0.2)."
        ),
    )

"""
Model insights and benchmark verification schemas.
"""
from typing import List, Optional
from pydantic import BaseModel, Field


class MetricSplit(BaseModel):
    """Verified performance metrics for a specific data split and threshold."""
    split_name: str = Field(..., example="Validation Split")
    sample_count: int = Field(..., example=4626)
    checkpoint: str = Field(..., example="best_fusion_model.zip (Epoch 7)")
    threshold: float = Field(..., example=0.41)
    roc_auc: float = Field(..., example=89.12)
    balanced_accuracy: float = Field(..., example=82.33)
    recall_sensitivity: float = Field(..., example=79.21)
    specificity: float = Field(..., example=85.45)
    precision: Optional[float] = Field(None, example=55.64)
    f1_score: Optional[float] = Field(None, example=65.36)


class BenchmarksResponse(BaseModel):
    """Complete verified benchmark evaluation results."""
    validation_metrics_at_041: MetricSplit
    validation_metrics_at_050: MetricSplit
    test_metrics_at_041: MetricSplit
    test_metrics_at_050: MetricSplit
    provenance_note: str = Field(
        ...,
        example="Metrics strictly reproduced from verified Epoch 7 checkpoint and cached test/val arrays.",
    )


class BranchArchitecture(BaseModel):
    """Architectural details for an individual feature branch."""
    name: str = Field(..., example="ResNet34-1D")
    role: str = Field(..., example="Morphological feature extraction")
    output_dimension: int = Field(..., example=512)
    details: str = Field(..., example="34-layer 1D CNN with residual blocks and adaptive pooling")


class ModelInfoResponse(BaseModel):
    """High-level architectural parameters and summary."""
    model_name: str = Field(..., example="CardioInsight ECGFusionClassifier")
    total_parameters: int = Field(..., example=2514819)
    trainable_parameters: int = Field(..., example=2514819)
    checkpoint_epoch: int = Field(..., example=7)
    branches: List[BranchArchitecture]
    fusion_mechanism: str = Field(
        ...,
        example="Adaptive Reliability Fusion with Softmax Gating & 8-Head Multi-Head Self-Attention in shared 512-d space",
    )
    classifier_head: str = Field(
        ...,
        example="4-stage MLP (512 -> 512 -> 256 -> 64 -> 1) with LayerNorm, GELU activations, and Dropout",
    )
    combined_system_parameters: Optional[int] = Field(
        10544669,
        example=10544669,
        description="Combined parameter count across ResNet34-1D (7,223,681), BiGRU (806,169), and Fusion module (2,514,819)",
    )

"""
Model insights and benchmark verification schemas.
"""
from typing import List, Optional
from pydantic import BaseModel, Field


class MetricSplit(BaseModel):
    """Verified performance metrics for a specific data split and threshold."""
    split_name: str = Field(..., example="Validation")
    sample_count: int = Field(..., example=4626)
    checkpoint: str = Field(..., example="Epoch 7 Best (best_fusion_model.zip)")
    threshold: float = Field(..., example=0.41)
    roc_auc: float = Field(..., example=88.92)
    balanced_accuracy: float = Field(..., example=82.35)
    recall_sensitivity: float = Field(..., example=77.94)
    specificity: float = Field(..., example=86.76)
    precision: Optional[float] = Field(None, example=57.54)
    f1_score: Optional[float] = Field(None, example=66.21)


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
        example="Adaptive Reliability Fusion with Softmax Gating & 8-Head Cross-Modal Attention",
    )
    classifier_head: str = Field(
        ...,
        example="3-stage MLP (512 -> 256 -> 64 -> 1) with LayerNorm and Dropout",
    )

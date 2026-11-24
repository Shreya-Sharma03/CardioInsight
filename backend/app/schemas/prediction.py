"""
Prediction request and response schemas for CardioInsight.
"""
from typing import List, Optional, Dict
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health check status and model availability."""
    status: str = Field(..., example="healthy")
    model_loaded: bool = Field(..., example=True)
    checkpoint_name: str = Field(..., example="best_fusion_model.zip (Epoch 7)")
    total_validation_records: int = Field(..., example=4626)
    device: str = Field(..., example="cpu")


class RecordPreset(BaseModel):
    """Suggested sample record metadata (no true label exposed)."""
    index: int = Field(..., example=42)
    label: str = Field(..., example="Record #0042")
    description: str = Field(..., example="Validation Cohort Sample #42")


class RecordsListResponse(BaseModel):
    """Available validation records summary and presets."""
    total_records: int = Field(..., example=4626)
    min_index: int = Field(..., example=0)
    max_index: int = Field(..., example=4625)
    suggested_records: List[RecordPreset]


class PredictRequest(BaseModel):
    """Inference request schema accepting a validation record index."""
    record_index: int = Field(
        ...,
        ge=0,
        le=4625,
        description="Zero-based index into validation feature arrays (0 to 4625)",
        example=42,
    )
    threshold: Optional[float] = Field(
        0.41,
        ge=0.0,
        le=1.0,
        description="Decision threshold for risk classification (default: 0.41)",
        example=0.41,
    )


class ReliabilityWeights(BaseModel):
    """Dynamic modality reliability weights computed by Adaptive Reliability Fusion gating."""
    resnet34: float = Field(..., description="ResNet34-1D morphological branch weight", example=0.5721)
    bigru: float = Field(..., description="BiGRU temporal dynamics branch weight", example=0.4279)
    weights_sum: float = Field(..., description="Sum of reliability weights (strictly ~1.0)", example=1.0000)


class EmbeddingsSummary(BaseModel):
    """Dimensions and Euclidean norms of the active latent feature vectors."""
    resnet_dim: int = Field(..., example=512)
    resnet_norm: float = Field(..., example=4.82)
    bigru_dim: int = Field(..., example=256)
    bigru_norm: float = Field(..., example=3.15)


class PredictResponse(BaseModel):
    """Complete prediction result without true label disclosure."""
    record_index: int = Field(..., example=42)
    probability: float = Field(..., description="Predicted probability of heart failure risk (0.0 to 1.0)", example=0.9412)
    risk_category: str = Field(..., description="Binary classification (HIGH RISK or LOW RISK)", example="HIGH RISK")
    is_high_risk: bool = Field(..., description="True if probability >= threshold", example=True)
    decision_threshold: float = Field(..., description="Threshold applied for risk classification", example=0.41)
    reliability_weights: ReliabilityWeights
    embeddings_summary: EmbeddingsSummary


class WaveformData(BaseModel):
    """Actual 12-lead ECG waveform signals and temporal metadata."""
    lead_names: List[str] = Field(..., description="List of 12 ECG lead identifiers in standard order")
    sampling_rate_hz: int = Field(250, description="Sampling rate in Hertz (250 Hz)")
    duration_seconds: float = Field(10.0, description="Total recording duration in seconds")
    total_samples: int = Field(..., description="Sample count per lead returned in payload")
    original_samples: int = Field(2500, description="Original full-resolution sample count per lead")
    time_axis: List[float] = Field(..., description="Time values from 0.0 to 10.0 seconds")
    leads: Dict[str, List[float]] = Field(..., description="Keyed dictionary of lead names to actual amplitude values")


class RawPredictResponse(BaseModel):
    """Prediction result including raw ECG inference metrics and actual 12-lead waveforms."""
    record_index: int = Field(..., example=42)
    probability: float = Field(..., description="Predicted probability of heart failure risk (0.0 to 1.0)", example=0.9499)
    risk_category: str = Field(..., description="Binary classification (HIGH RISK or LOW RISK)", example="HIGH RISK")
    is_high_risk: bool = Field(..., description="True if probability >= threshold", example=True)
    decision_threshold: float = Field(..., description="Threshold applied for risk classification", example=0.41)
    reliability_weights: ReliabilityWeights
    embeddings_summary: EmbeddingsSummary
    waveform: WaveformData
    inference_source: str = Field("raw_ecg_backbone_fusion", description="Source of feature extraction and inference")

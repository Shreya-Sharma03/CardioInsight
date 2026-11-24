"""
Prediction and record retrieval API routes.
"""
from fastapi import APIRouter, HTTPException, status
from backend.app.schemas.prediction import (
    HealthResponse,
    RecordsListResponse,
    PredictRequest,
    PredictResponse,
    RawPredictResponse,
)
from backend.app.services.model_service import get_model_service

router = APIRouter(tags=["Prediction"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check and model load status",
)
def get_health():
    """Verify backend health and confirms that the fusion model checkpoint is loaded."""
    service = get_model_service()
    return HealthResponse(
        status="healthy" if service.is_loaded else "uninitialized",
        model_loaded=service.is_loaded,
        checkpoint_name="best_fusion_model.zip (Epoch 7)",
        total_validation_records=service.total_records,
        device=str(service.device),
    )


@router.get(
    "/records",
    response_model=RecordsListResponse,
    summary="List validation cohort records and suggested presets",
)
def list_records():
    """
    Retrieve summary of available validation records (0 to 4625) and suggested presets.
    Ground-truth labels are strictly withheld.
    """
    service = get_model_service()
    if not service.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model service is not loaded yet.",
        )
    return RecordsListResponse(**service.get_records_summary())


@router.post(
    "/predict",
    response_model=PredictResponse,
    summary="Run PyTorch inference on a validation record",
)
def predict(request: PredictRequest):
    """
    Execute real model inference using the verified best fusion model checkpoint.

    - Validates that record_index is within bounds [0, 4625].
    - Retrieves the precomputed ResNet34 (512-d) and BiGRU (256-d) feature vectors.
    - Runs a forward pass under torch.no_grad().
    - Returns risk probability, classification at threshold 0.41, and dynamic fusion weights.
    - True validation labels are not exposed.
    """
    service = get_model_service()
    if not service.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model service is not loaded yet.",
        )

    try:
        threshold = request.threshold if request.threshold is not None else 0.41
        result = service.predict_record(
            record_index=request.record_index,
            threshold=threshold,
        )
        return PredictResponse(**result)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}",
        )


@router.post(
    "/predict/raw-record",
    response_model=RawPredictResponse,
    summary="Run genuine end-to-end inference on actual EchoNext raw ECG waveform",
)
def predict_raw_record(request: PredictRequest):
    """
    Execute real end-to-end model inference using actual EchoNext validation ECG waveforms:
    1. Validates record_index [0, 4625].
    2. Retrieves genuine 12-lead waveform from data/raw/EchoNext_val_waveforms.npy via mmap_mode='r'.
    3. Transforms tensor to (1, 12, 2500) float32.
    4. Extracts ResNet34-1D 512-dim features.
    5. Extracts BiGRU 256-dim features.
    6. Passes both into the trained Adaptive Reliability Fusion model.
    7. Calculates sigmoid probability and classifies against threshold.
    8. Returns the prediction and the actual 12-lead ECG waveform signals for time 0 to 10s.
    """
    service = get_model_service()
    if not service.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model service is not loaded yet.",
        )

    try:
        threshold = request.threshold if request.threshold is not None else 0.41
        result = service.predict_raw_record(
            record_index=request.record_index,
            threshold=threshold,
            downsample_factor=2,  # 1250 points per lead for optimal web rendering performance
        )
        return RawPredictResponse(**result)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Raw ECG inference error: {str(e)}",
        )

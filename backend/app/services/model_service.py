"""
Model inference and data service for CardioInsight.

Encapsulates:
- Loading the verified best fusion model checkpoint (Epoch 7)
- Loading trained ResNet34-1D and BiGRU backbone models strictly
- Accessing actual EchoNext validation waveforms via mmap_mode='r'
- Executing end-to-end PyTorch raw ECG inference under torch.no_grad()
- Retaining backward-compatible cached precomputed feature inference
"""
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List

import numpy as np
import torch

# Ensure project root is in sys.path so existing modular packages can be imported
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.evaluator import load_fusion_checkpoint
from resnet_model import ResNet34_1D
from bigru_model import BiGRUECGClassifier
from config.config import DEFAULT_PATH_CONFIG, DEFAULT_EVAL_CONFIG

# Standard 12-lead names in physiological order
ECG_LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


class ModelService:
    """Singleton service for model loading and real-time inference."""

    def __init__(self, checkpoint_path: Optional[str] = None):
        self.project_root = PROJECT_ROOT
        self.checkpoint_path = checkpoint_path or str(self.project_root / DEFAULT_PATH_CONFIG.best_fusion_checkpoint)
        self.device = torch.device("cpu")

        # Models
        self.fusion_model = None
        self.resnet_model = None
        self.bigru_model = None

        # Feature caches & raw waveforms
        self.resnet_val = None
        self.bigru_val = None
        self.raw_waveforms = None  # mmap_mode='r'

        self.total_records = 0
        self.is_loaded = False

    def load(self):
        """Load all model checkpoints once during startup and open memory-mapped raw waveforms."""
        if self.is_loaded:
            return

        # 1. Load strict fusion model checkpoint
        resolved_fusion_ckpt = Path(self.checkpoint_path)
        if not resolved_fusion_ckpt.exists():
            resolved_fusion_ckpt = self.project_root / "best_fusion_model.zip"

        if not resolved_fusion_ckpt.exists():
            raise FileNotFoundError(f"Fusion checkpoint file not found: {resolved_fusion_ckpt}")

        self.fusion_model = load_fusion_checkpoint(str(resolved_fusion_ckpt), device=self.device, strict=True)
        self.fusion_model.eval()

        # 2. Load ResNet34-1D Backbone Model
        resnet_ckpt_path = self.project_root / DEFAULT_PATH_CONFIG.resnet_checkpoint
        if not resnet_ckpt_path.exists():
            resnet_ckpt_path = self.project_root / "ResNet34_ECG.pth.zip"
        if not resnet_ckpt_path.exists():
            raise FileNotFoundError(f"ResNet34 checkpoint file not found: {resnet_ckpt_path}")

        self.resnet_model = ResNet34_1D()
        ckpt_res = torch.load(str(resnet_ckpt_path), map_location=self.device, weights_only=False)
        state_res = ckpt_res["model"] if isinstance(ckpt_res, dict) and "model" in ckpt_res else ckpt_res
        self.resnet_model.load_state_dict(state_res, strict=True)
        self.resnet_model.to(self.device)
        self.resnet_model.eval()

        # 3. Load BiGRU Backbone Model
        bigru_ckpt_path = self.project_root / DEFAULT_PATH_CONFIG.bigru_checkpoint
        if not bigru_ckpt_path.exists():
            bigru_ckpt_path = self.project_root / "checkpoint_backup" / "BiGRU_ECG_repacked.zip"
        if not bigru_ckpt_path.exists():
            raise FileNotFoundError(f"BiGRU checkpoint file not found: {bigru_ckpt_path}")

        self.bigru_model = BiGRUECGClassifier()
        ckpt_gru = torch.load(str(bigru_ckpt_path), map_location=self.device, weights_only=False)
        state_gru = ckpt_gru["model"] if isinstance(ckpt_gru, dict) and "model" in ckpt_gru else ckpt_gru
        self.bigru_model.load_state_dict(state_gru, strict=True)
        self.bigru_model.to(self.device)
        self.bigru_model.eval()

        # 4. Memory-map raw validation waveforms (does not load entire ~1 GB file into RAM)
        raw_waveforms_path = self.project_root / DEFAULT_PATH_CONFIG.val_waveforms_file
        if not raw_waveforms_path.exists():
            raw_waveforms_path = self.project_root / "data" / "raw" / "EchoNext_val_waveforms.npy"
        if not raw_waveforms_path.exists():
            raise FileNotFoundError(f"EchoNext validation waveform file not found: {raw_waveforms_path}")

        self.raw_waveforms = np.load(str(raw_waveforms_path), mmap_mode="r")

        # 5. Load cached validation feature matrices for legacy /api/predict compatibility
        val_res_path = self.project_root / DEFAULT_PATH_CONFIG.resnet_features_val
        val_gru_path = self.project_root / DEFAULT_PATH_CONFIG.bigru_features_val

        if not val_res_path.exists():
            val_res_path = self.project_root / "resnet_features_val.npy"
        if not val_gru_path.exists():
            val_gru_path = self.project_root / "bigru_features_val.npy"

        self.resnet_val = np.load(str(val_res_path))
        self.bigru_val = np.load(str(val_gru_path))

        assert len(self.raw_waveforms) == len(self.resnet_val) == len(self.bigru_val), (
            f"Record count mismatch: raw={len(self.raw_waveforms)}, "
            f"ResNet={len(self.resnet_val)}, BiGRU={len(self.bigru_val)}"
        )
        self.total_records = len(self.raw_waveforms)
        self.is_loaded = True

    @property
    def model(self):
        """Backward-compatible alias for fusion_model."""
        return self.fusion_model

    def get_records_summary(self) -> Dict[str, Any]:
        """Return available record counts and suggested presets without true labels."""
        if not self.is_loaded:
            raise RuntimeError("ModelService is not loaded.")

        suggested = [
            {"index": 42, "label": "Record #0042", "description": "Validation Cohort Sample #42 (High Risk)"},
            {"index": 0, "label": "Record #0000", "description": "Validation Cohort Sample #0 (Low Risk)"},
            {"index": 100, "label": "Record #0100", "description": "Validation Cohort Sample #100"},
            {"index": 1000, "label": "Record #1000", "description": "Validation Cohort Sample #1000"},
            {"index": 4542, "label": "Record #4542", "description": "Validation Cohort Sample #4542 (Boundary Case ~0.41)"},
        ]

        return {
            "total_records": self.total_records,
            "min_index": 0,
            "max_index": self.total_records - 1,
            "suggested_records": suggested,
        }

    def predict_record(self, record_index: int, threshold: float = 0.41) -> Dict[str, Any]:
        """
        Run genuine PyTorch inference using precomputed cached feature vectors (Legacy API).
        """
        if not self.is_loaded:
            raise RuntimeError("ModelService is not loaded.")

        if not (0 <= record_index < self.total_records):
            raise ValueError(f"Record index {record_index} is out of bounds [0, {self.total_records - 1}].")

        res_vec = self.resnet_val[record_index]  # (512,)
        gru_vec = self.bigru_val[record_index]  # (256,)

        res_tensor = torch.tensor(res_vec, dtype=torch.float32).unsqueeze(0).to(self.device)
        gru_tensor = torch.tensor(gru_vec, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits, fused, alpha_res, alpha_gru = self.fusion_model(res_tensor, gru_tensor)
            prob = float(torch.sigmoid(logits).squeeze().item())
            w_resnet = float(alpha_res.squeeze().item())
            w_bigru = float(alpha_gru.squeeze().item())

        is_high_risk = prob >= threshold
        risk_category = "HIGH RISK" if is_high_risk else "LOW RISK"

        return {
            "record_index": record_index,
            "probability": round(prob, 4),
            "risk_category": risk_category,
            "is_high_risk": is_high_risk,
            "decision_threshold": round(threshold, 4),
            "reliability_weights": {
                "resnet34": round(w_resnet, 4),
                "bigru": round(w_bigru, 4),
                "weights_sum": round(w_resnet + w_bigru, 4),
            },
            "embeddings_summary": {
                "resnet_dim": int(res_vec.shape[0]),
                "resnet_norm": round(float(np.linalg.norm(res_vec)), 2),
                "bigru_dim": int(gru_vec.shape[0]),
                "bigru_norm": round(float(np.linalg.norm(gru_vec)), 2),
            },
        }

    def predict_raw_record(
        self,
        record_index: int,
        threshold: float = 0.41,
        downsample_factor: int = 2,
    ) -> Dict[str, Any]:
        """
        Run genuine end-to-end inference directly on the actual raw EchoNext ECG waveform.

        Flow:
        1. Retrieve record_index from memory-mapped EchoNext_val_waveforms.npy: shape (1, 2500, 12).
        2. Transmute shape to (1, 12, 2500) as float32 for model inputs.
        3. Pass waveform into ResNet34-1D to extract 512-dim latent vector.
        4. Pass same waveform into BiGRU to extract 256-dim latent vector.
        5. Pass both extracted vectors into Adaptive Reliability Fusion classifier.
        6. Apply sigmoid to compute risk probability and classify against threshold.
        7. Extract actual 12-lead signal data (2500 samples over 10.0 seconds) for frontend visualization.
        """
        if not self.is_loaded:
            raise RuntimeError("ModelService is not loaded.")

        if not (0 <= record_index < self.total_records):
            raise ValueError(f"Record index {record_index} is out of bounds [0, {self.total_records - 1}].")

        # 1. Retrieve raw record via memory map (1, 2500, 12)
        raw_slice = np.array(self.raw_waveforms[record_index], copy=True)  # (1, 2500, 12) float64

        # 2. Convert to PyTorch float32 tensor of shape (1, 12, 2500)
        # Permute (batch=0, time=1, lead=2) -> (batch=0, lead=2, time=1)
        wf_tensor = torch.from_numpy(raw_slice).permute(0, 2, 1).to(device=self.device, dtype=torch.float32)

        # 3. Model forward pass through ResNet34, BiGRU, and Fusion
        with torch.no_grad():
            # ResNet34-1D returns: (logit, hidden_features [1, 512])
            _, res_feat = self.resnet_model(wf_tensor)

            # BiGRU returns: (logit, hidden_features [1, 256])
            _, gru_feat = self.bigru_model(wf_tensor)

            # Fusion Classifier returns: (logits, fused, alpha_res, alpha_gru)
            logits, fused, alpha_res, alpha_gru = self.fusion_model(res_feat, gru_feat)
            prob = float(torch.sigmoid(logits).squeeze().item())
            w_resnet = float(alpha_res.squeeze().item())
            w_bigru = float(alpha_gru.squeeze().item())

        is_high_risk = prob >= threshold
        risk_category = "HIGH RISK" if is_high_risk else "LOW RISK"

        # 4. Format 12-lead waveforms for API response
        # Shape of raw_slice is (1, 2500, 12) -> squeeze(0) is (2500, 12)
        waveform_data = raw_slice[0]  # (2500, 12)
        total_samples = waveform_data.shape[0]  # 2500
        sampling_rate = 250.0  # 250 Hz (2500 samples / 10s)
        time_step = 1.0 / sampling_rate  # 0.004s

        # Downsample strictly for response/display payload if requested (model inference used full 2500)
        step = max(1, int(downsample_factor))
        time_axis = [round(float(i * time_step), 4) for i in range(0, total_samples, step)]

        leads_dict: Dict[str, List[float]] = {}
        for lead_idx, lead_name in enumerate(ECG_LEAD_NAMES):
            lead_samples = waveform_data[::step, lead_idx]
            leads_dict[lead_name] = [round(float(val), 4) for val in lead_samples]

        res_np = res_feat.squeeze(0).cpu().numpy()
        gru_np = gru_feat.squeeze(0).cpu().numpy()

        return {
            "record_index": record_index,
            "probability": round(prob, 4),
            "risk_category": risk_category,
            "is_high_risk": is_high_risk,
            "decision_threshold": round(threshold, 4),
            "reliability_weights": {
                "resnet34": round(w_resnet, 4),
                "bigru": round(w_bigru, 4),
                "weights_sum": round(w_resnet + w_bigru, 4),
            },
            "embeddings_summary": {
                "resnet_dim": int(res_np.shape[0]),
                "resnet_norm": round(float(np.linalg.norm(res_np)), 2),
                "bigru_dim": int(gru_np.shape[0]),
                "bigru_norm": round(float(np.linalg.norm(gru_np)), 2),
            },
            "waveform": {
                "lead_names": ECG_LEAD_NAMES,
                "sampling_rate_hz": 250,
                "duration_seconds": 10.0,
                "total_samples": len(time_axis),
                "original_samples": total_samples,
                "time_axis": time_axis,
                "leads": leads_dict,
            },
            "inference_source": "raw_ecg_backbone_fusion",
        }


# Global singleton instance
_model_service_instance: Optional[ModelService] = None


def get_model_service() -> ModelService:
    """Retrieve or initialize the global ModelService instance."""
    global _model_service_instance
    if _model_service_instance is None:
        _model_service_instance = ModelService()
    return _model_service_instance

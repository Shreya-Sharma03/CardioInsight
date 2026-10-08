"""
Inference API and CLI for CardioInsight.

Supports end-to-end heart failure risk prediction from raw 12-lead ECG signals
or pre-extracted morphological and temporal feature representations.
"""

import os
import argparse
from typing import Dict, Any, Union
import numpy as np
import torch

try:
    from .resnet_model import ResNet34_1D
    from .bigru_model import BiGRUECGClassifier
    from .fusion_model import ECGFusionClassifier
    from .preprocessing import preprocess_ecg
except ImportError:
    from resnet_model import ResNet34_1D
    from bigru_model import BiGRUECGClassifier
    from fusion_model import ECGFusionClassifier
    from preprocessing import preprocess_ecg


class CardioInsightPredictor:
    """
    Unified predictor executing the dual-branch ResNet34-1D + BiGRU + Fusion pipeline.
    """

    def __init__(
        self,
        models_dir: str = None,
        resnet_ckpt: str = None,
        bigru_ckpt: str = None,
        fusion_ckpt: str = None,
        threshold: float = None,
        device: str = "cpu",
    ):
        self.device = torch.device(device)

        if models_dir is None:
            # Default to ../models relative to this file
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            models_dir = os.path.join(base_dir, "models")

        self.resnet_ckpt = resnet_ckpt or os.path.join(models_dir, "ResNet34_ECG.pth")
        self.bigru_ckpt = bigru_ckpt or os.path.join(models_dir, "BiGRU_ECG.pth")
        self.fusion_ckpt = fusion_ckpt or os.path.join(models_dir, "FusionModel.pth")
        
        # Load threshold
        if threshold is not None:
            self.threshold = threshold
        else:
            thresh_path = os.path.join(models_dir, "threshold.npy")
            if os.path.exists(thresh_path):
                self.threshold = float(np.load(thresh_path))
            else:
                self.threshold = 0.41  # Optimal threshold determined on validation set

        # Load label map
        label_map_path = os.path.join(models_dir, "labels.pth")
        if os.path.exists(label_map_path):
            self.label_map = torch.load(label_map_path, map_location="cpu")
        else:
            self.label_map = {0: "Low Risk", 1: "High Risk"}

        # Initialize models
        self.resnet = ResNet34_1D().to(self.device)
        self.bigru = BiGRUECGClassifier().to(self.device)
        self.fusion = ECGFusionClassifier().to(self.device)

        self._load_checkpoints()
        self.resnet.eval()
        self.bigru.eval()
        self.fusion.eval()

    def _load_checkpoints(self):
        """Load state dicts if checkpoint files exist."""
        if os.path.exists(self.resnet_ckpt):
            self.resnet.load_state_dict(
                torch.load(self.resnet_ckpt, map_location=self.device), strict=True
            )
        else:
            print(f"[Warning] ResNet checkpoint not found at {self.resnet_ckpt}")

        if os.path.exists(self.bigru_ckpt):
            self.bigru.load_state_dict(
                torch.load(self.bigru_ckpt, map_location=self.device), strict=True
            )
        else:
            print(f"[Warning] BiGRU checkpoint not found at {self.bigru_ckpt}")

        if os.path.exists(self.fusion_ckpt):
            self.fusion.load_state_dict(
                torch.load(self.fusion_ckpt, map_location=self.device), strict=True
            )
        else:
            print(f"[Warning] Fusion checkpoint not found at {self.fusion_ckpt}")

    def predict_ecg(
        self,
        ecg_signal: Union[np.ndarray, list],
        fs: float = 250.0,
        apply_filter: bool = True,
    ) -> Dict[str, Any]:
        """
        Run end-to-end risk stratification from raw 12-lead ECG.

        Parameters
        ----------
        ecg_signal : array-like of shape (12, T) or (T, 12)
            Multi-lead raw ECG.
        fs : float
            Sampling rate (default 250 Hz).
        apply_filter : bool
            Whether to apply Butterworth bandpass filtering.

        Returns
        -------
        result : dict
            Contains probability, predicted_class, risk_label, modality weights.
        """
        # Preprocessing
        proc_ecg = preprocess_ecg(ecg_signal, fs=fs, apply_filter=apply_filter)
        ecg_tensor = torch.from_numpy(proc_ecg).unsqueeze(0).to(self.device)  # (1, 12, 2500)

        with torch.no_grad():
            res_logit, res_feat = self.resnet(ecg_tensor)
            gru_logit, gru_feat = self.bigru(ecg_tensor)
            fusion_logit, fused_feat, alpha_res, alpha_gru = self.fusion(res_feat, gru_feat)

            res_prob = torch.sigmoid(res_logit).item()
            gru_prob = torch.sigmoid(gru_logit).item()
            fusion_prob = torch.sigmoid(fusion_logit).item()

        binary_pred = int(fusion_prob >= self.threshold)
        risk_label = self.label_map.get(binary_pred, "Unknown")

        return {
            "heart_failure_risk_probability": round(fusion_prob, 4),
            "binary_prediction": binary_pred,
            "risk_stratification": risk_label,
            "decision_threshold": self.threshold,
            "modality_reliability_weights": {
                "resnet_morphological_weight": round(alpha_res.item(), 4),
                "bigru_temporal_weight": round(alpha_gru.item(), 4),
            },
            "branch_probabilities": {
                "resnet_alone": round(res_prob, 4),
                "bigru_alone": round(gru_prob, 4),
            },
        }

    def predict_features(
        self,
        res_feat: np.ndarray,
        gru_feat: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Predict heart failure risk directly from extracted features.
        """
        res_t = torch.from_numpy(res_feat).float().to(self.device)
        gru_t = torch.from_numpy(gru_feat).float().to(self.device)
        if res_t.ndim == 1:
            res_t = res_t.unsqueeze(0)
        if gru_t.ndim == 1:
            gru_t = gru_t.unsqueeze(0)

        with torch.no_grad():
            fusion_logit, fused_feat, alpha_res, alpha_gru = self.fusion(res_t, gru_t)
            probs = torch.sigmoid(fusion_logit).cpu().numpy()

        binary_preds = (probs >= self.threshold).astype(int)
        
        return {
            "probabilities": probs,
            "predictions": binary_preds,
            "alpha_res": alpha_res.cpu().numpy(),
            "alpha_gru": alpha_gru.cpu().numpy(),
            "threshold": self.threshold,
        }


def predict_risk(ecg_signal: np.ndarray, **kwargs) -> Dict[str, Any]:
    """Convenience helper function for fast inference."""
    predictor = CardioInsightPredictor(**kwargs)
    return predictor.predict_ecg(ecg_signal)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CardioInsight Heart Failure Risk Prediction")
    parser.add_argument("--demo", action="store_true", help="Run quick demo on synthetic 12-lead ECG")
    parser.add_argument("--ecg_file", type=str, default=None, help="Path to .npy 12-lead ECG array")
    parser.add_argument("--threshold", type=float, default=None, help="Decision threshold override")
    args = parser.parse_args()

    if args.demo or args.ecg_file is None:
        print("\n--- Running CardioInsight Synthetic ECG Inference Demo ---")
        synthetic_ecg = np.random.randn(12, 2500).astype(np.float32)
        predictor = CardioInsightPredictor(threshold=args.threshold)
        res = predictor.predict_ecg(synthetic_ecg)
        print("Prediction Result:")
        for k, v in res.items():
            print(f"  {k}: {v}")
    else:
        ecg_data = np.load(args.ecg_file)
        predictor = CardioInsightPredictor(threshold=args.threshold)
        res = predictor.predict_ecg(ecg_data)
        print("Prediction Result:")
        for k, v in res.items():
            print(f"  {k}: {v}")

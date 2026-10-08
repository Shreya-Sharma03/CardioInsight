"""
Comprehensive Test Suite for CardioInsight.

Validates model architectures, checkpoint loading, preprocessing,
end-to-end inference, and clinical evaluation metrics.
"""

import os
import sys
from pathlib import Path
import unittest
import numpy as np
import torch

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.resnet_model import ResNet34_1D
from src.bigru_model import BiGRUECGClassifier
from src.fusion_model import ECGFusionClassifier, AdaptiveReliabilityFusion
from src.preprocessing import preprocess_ecg, bandpass_filter, normalize_ecg
from src.predict import CardioInsightPredictor
from src.evaluate import compute_clinical_metrics


class TestCardioInsightPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.models_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"
        )
        cls.resnet_ckpt = os.path.join(cls.models_dir, "ResNet34_ECG.pth")
        cls.bigru_ckpt = os.path.join(cls.models_dir, "BiGRU_ECG.pth")
        cls.fusion_ckpt = os.path.join(cls.models_dir, "FusionModel.pth")

    def test_01_resnet_architecture(self):
        """Test ResNet34-1D output shapes and feature dimension."""
        model = ResNet34_1D()
        x = torch.randn(2, 12, 2500)
        logit, feat = model(x)
        self.assertEqual(logit.shape, torch.Size([2]))
        self.assertEqual(feat.shape, torch.Size([2, 512]))

    def test_02_bigru_architecture(self):
        """Test BiGRUECGClassifier output shapes and feature dimension."""
        model = BiGRUECGClassifier()
        x = torch.randn(2, 12, 2500)
        logit, feat = model(x)
        self.assertEqual(logit.shape, torch.Size([2]))
        self.assertEqual(feat.shape, torch.Size([2, 256]))

    def test_03_fusion_architecture(self):
        """Test AdaptiveReliabilityFusion and ECGFusionClassifier."""
        model = ECGFusionClassifier()
        res_feat = torch.randn(4, 512)
        gru_feat = torch.randn(4, 256)
        logit, fused, alpha_res, alpha_gru = model(res_feat, gru_feat)

        self.assertEqual(logit.shape, torch.Size([4]))
        self.assertEqual(fused.shape, torch.Size([4, 512]))
        self.assertEqual(alpha_res.shape, torch.Size([4, 1]))
        self.assertEqual(alpha_gru.shape, torch.Size([4, 1]))

        # Reliability weights must sum to 1.0 for each sample
        weight_sum = (alpha_res + alpha_gru).squeeze(-1)
        np.testing.assert_allclose(weight_sum.detach().numpy(), np.ones(4), rtol=1e-5)

    def test_04_resnet_checkpoint_loading(self):
        """Verify ResNet34 checkpoint loads strictly without missing/unexpected keys."""
        if os.path.exists(self.resnet_ckpt):
            model = ResNet34_1D()
            sd = torch.load(self.resnet_ckpt, map_location="cpu")
            missing, unexpected = model.load_state_dict(sd, strict=True)
            self.assertEqual(len(missing), 0)
            self.assertEqual(len(unexpected), 0)

    def test_05_bigru_checkpoint_loading(self):
        """Verify BiGRU checkpoint loads strictly without missing/unexpected keys."""
        if os.path.exists(self.bigru_ckpt):
            model = BiGRUECGClassifier()
            sd = torch.load(self.bigru_ckpt, map_location="cpu")
            missing, unexpected = model.load_state_dict(sd, strict=True)
            self.assertEqual(len(missing), 0)
            self.assertEqual(len(unexpected), 0)

    def test_06_fusion_checkpoint_loading(self):
        """Verify Fusion checkpoint loads strictly."""
        if os.path.exists(self.fusion_ckpt):
            model = ECGFusionClassifier()
            sd = torch.load(self.fusion_ckpt, map_location="cpu")
            missing, unexpected = model.load_state_dict(sd, strict=True)
            self.assertEqual(len(missing), 0)
            self.assertEqual(len(unexpected), 0)

    def test_07_preprocessing(self):
        """Test filtering, normalization, and shape alignment."""
        raw = np.random.randn(12, 3000).astype(np.float32)
        proc = preprocess_ecg(raw, fs=250.0, target_length=2500)
        self.assertEqual(proc.shape, (12, 2500))

        # Test transposed input
        transposed = np.random.randn(2500, 12).astype(np.float32)
        proc_t = preprocess_ecg(transposed, fs=250.0, target_length=2500)
        self.assertEqual(proc_t.shape, (12, 2500))

    def test_08_end_to_end_predictor(self):
        """Test full inference pipeline from raw ECG signal."""
        predictor = CardioInsightPredictor(models_dir=self.models_dir)
        raw_ecg = np.random.randn(12, 2500).astype(np.float32)
        res = predictor.predict_ecg(raw_ecg)

        self.assertIn("heart_failure_risk_probability", res)
        self.assertIn("risk_stratification", res)
        self.assertIn(res["risk_stratification"], ["Low Risk", "High Risk"])
        self.assertIn("modality_reliability_weights", res)

        w_res = res["modality_reliability_weights"]["resnet_morphological_weight"]
        w_gru = res["modality_reliability_weights"]["bigru_temporal_weight"]
        self.assertAlmostEqual(w_res + w_gru, 1.0, places=3)

    def test_09_metrics_computation(self):
        """Test clinical metrics calculation."""
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_probs = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7])
        m = compute_clinical_metrics(y_true, y_probs, threshold=0.5)

        self.assertEqual(m["accuracy"], 1.0)
        self.assertEqual(m["balanced_accuracy"], 1.0)
        self.assertEqual(m["f1_score"], 1.0)
        self.assertEqual(m["roc_auc"], 1.0)


if __name__ == "__main__":
    unittest.main()

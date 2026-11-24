/**
 * CardioInsight TypeScript Interface Definitions
 * Synchronized with FastAPI Backend Pydantic Schemas
 */

export interface HealthResponse {
  status: string;
  model_loaded: boolean;
  checkpoint_name: string;
  total_validation_records: number;
  device: string;
}

export interface RecordPreset {
  index: number;
  label: string;
  description: string;
}

export interface RecordsListResponse {
  total_records: number;
  min_index: number;
  max_index: number;
  suggested_records: RecordPreset[];
}

export interface PredictRequest {
  record_index: number;
  threshold?: number;
}

export interface ReliabilityWeights {
  resnet34: number;
  bigru: number;
  weights_sum: number;
}

export interface EmbeddingsSummary {
  resnet_dim: number;
  resnet_norm: number;
  bigru_dim: number;
  bigru_norm: number;
}

export interface PredictResponse {
  record_index: number;
  probability: number;
  risk_category: 'HIGH RISK' | 'LOW RISK';
  is_high_risk: boolean;
  decision_threshold: number;
  reliability_weights: ReliabilityWeights;
  embeddings_summary: EmbeddingsSummary;
}

export interface WaveformData {
  lead_names: string[];
  sampling_rate_hz: number;
  duration_seconds: number;
  total_samples: number;
  original_samples: number;
  time_axis: number[];
  leads: Record<string, number[]>;
}

export interface RawPredictResponse {
  record_index: number;
  probability: number;
  risk_category: 'HIGH RISK' | 'LOW RISK';
  is_high_risk: boolean;
  decision_threshold: number;
  reliability_weights: ReliabilityWeights;
  embeddings_summary: EmbeddingsSummary;
  waveform: WaveformData;
  inference_source: string;
}

export interface MetricSplit {
  split_name: string;
  sample_count: number;
  checkpoint: string;
  threshold: number;
  roc_auc: number;
  balanced_accuracy: number;
  recall_sensitivity: number;
  specificity: number;
  precision?: number;
  f1_score?: number;
}

export interface BenchmarksResponse {
  validation_metrics_at_041: MetricSplit;
  validation_metrics_at_050: MetricSplit;
  test_metrics_at_041: MetricSplit;
  test_metrics_at_050: MetricSplit;
  provenance_note: string;
}

export interface BranchArchitecture {
  name: string;
  role: string;
  output_dimension: number;
  details: string;
}

export interface ModelInfoResponse {
  model_name: string;
  total_parameters: number;
  trainable_parameters: number;
  checkpoint_epoch: number;
  branches: BranchArchitecture[];
  fusion_mechanism: string;
  classifier_head: string;
}

export type ActivePage = 'home' | 'prediction' | 'insights';

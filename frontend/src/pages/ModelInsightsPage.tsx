import React, { useState, useEffect } from 'react';
import { Layers, Sliders, AlertTriangle, ArrowRight, ArrowDown, Cpu, Sparkles, Database } from 'lucide-react';
import { fetchBenchmarks, fetchModelInfo } from '../api/client';
import { BenchmarksResponse, ModelInfoResponse } from '../types';
import { ConfusionMatrixPlot } from '../components/ConfusionMatrixPlot';
import { MetricsComparisonPlot } from '../components/MetricsComparisonPlot';

export const ModelInsightsPage: React.FC = () => {
  const [benchmarks, setBenchmarks] = useState<BenchmarksResponse | null>(null);
  const [modelInfo, setModelInfo] = useState<ModelInfoResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);

  useEffect(() => {
    let isMounted = true;
    async function loadData() {
      try {
        const [bData, mData] = await Promise.all([fetchBenchmarks(), fetchModelInfo()]);
        if (isMounted) {
          setBenchmarks(bData);
          setModelInfo(mData);
        }
      } catch (err) {
        console.error('Failed to load insights data:', err);
      } finally {
        if (isMounted) setLoading(false);
      }
    }
    loadData();
    return () => {
      isMounted = false;
    };
  }, []);

  return (
    <div className="space-y-12 py-6">
      {/* Page Header */}
      <div className="pb-4 border-b border-sandBorder">
        <h1 className="text-2xl sm:text-3xl font-extrabold text-charcoal tracking-tight">
          Model Architecture & Performance Insights
        </h1>
        <p className="text-xs sm:text-sm text-charcoal-muted mt-1">
          Detailed technical audit, verified evaluation benchmarks, and cross-modal fusion mechanics.
        </p>
      </div>

      {/* Architecture Flow Diagram */}
      <section className="space-y-6">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Layers className="w-5 h-5 text-sage" />
            <h2 className="text-lg font-bold text-charcoal">
              Adaptive Reliability Fusion Architecture
            </h2>
          </div>
          <span className="text-xs font-mono bg-sand border border-sandBorder px-3 py-1 rounded-md text-charcoal">
            {modelInfo ? `${modelInfo.total_parameters.toLocaleString()} Total Parameters` : '2,514,819 Parameters'}
          </span>
        </div>

        {/* Visual Pipeline Layout with Clear Directional Flow */}
        <div className="bg-surface rounded-xl border border-sandBorder p-6 shadow-sm space-y-6">
          <div className="flex flex-col lg:flex-row items-stretch lg:items-center justify-between gap-4">

            {/* Stage 1: Dual Input Branches */}
            <div className="flex-1 flex flex-col gap-3 min-w-[220px]">
              <div className="text-[11px] font-semibold uppercase tracking-wider text-charcoal-muted text-center lg:text-left">
                Stage 1: Input Branches
              </div>

              {/* ResNet34 Branch */}
              <div className="p-3.5 rounded-lg bg-sand border border-sandBorder space-y-1 relative group hover:border-sage transition-colors">
                <div className="flex items-center justify-between">
                  <span className="font-bold text-xs text-charcoal flex items-center gap-1.5">
                    <span className="w-2 h-2 rounded-full bg-sage"></span>
                    ResNet34-1D Branch
                  </span>
                  <span className="text-[10px] font-mono font-semibold px-1.5 py-0.5 bg-white rounded border border-sandBorder text-charcoal">512-d</span>
                </div>
                <p className="text-[11px] text-charcoal-muted leading-tight">
                  Morphological waveform features extracted via 34-layer residual conv blocks.
                </p>
              </div>

              {/* BiGRU Branch */}
              <div className="p-3.5 rounded-lg bg-sand border border-sandBorder space-y-1 relative group hover:border-terracotta transition-colors">
                <div className="flex items-center justify-between">
                  <span className="font-bold text-xs text-charcoal flex items-center gap-1.5">
                    <span className="w-2 h-2 rounded-full bg-terracotta"></span>
                    BiGRU Temporal Branch
                  </span>
                  <span className="text-[10px] font-mono font-semibold px-1.5 py-0.5 bg-white rounded border border-sandBorder text-charcoal">256-d</span>
                </div>
                <p className="text-[11px] text-charcoal-muted leading-tight">
                  Temporal cadence & sequence dynamics from 3-layer bidirectional recurrent network.
                </p>
              </div>
            </div>

            {/* Connector Arrow 1 */}
            <div className="flex items-center justify-center text-charcoal-muted/60 py-1 lg:py-0">
              <ArrowRight className="hidden lg:block w-5 h-5 text-sage" />
              <ArrowDown className="block lg:hidden w-5 h-5 text-sage" />
            </div>

            {/* Stage 2: Projection & Reliability Gating */}
            <div className="flex-1 min-w-[220px] flex flex-col justify-center">
              <div className="text-[11px] font-semibold uppercase tracking-wider text-charcoal-muted text-center mb-3">
                Stage 2: Reliability Gating
              </div>
              <div className="p-4 rounded-lg bg-sand border border-sandBorder space-y-2 text-center h-full flex flex-col justify-center">
                <div className="flex justify-center">
                  <div className="inline-flex p-2 rounded-full bg-white text-sage shadow-xs">
                    <Sliders className="w-4 h-4" />
                  </div>
                </div>
                <h4 className="font-bold text-xs text-charcoal">
                  Softmax Gating Mechanism
                </h4>
                <p className="text-[11px] text-charcoal-muted leading-tight">
                  Projects features into unified 512-d space and estimates sample-specific confidence weights:
                </p>
                <div className="pt-1">
                  <code className="text-[10px] bg-white px-2 py-1 rounded font-mono text-sage-dark border border-sage-border inline-block shadow-2xs font-semibold">
                    &alpha;<sub>res</sub> + &alpha;<sub>gru</sub> = 1.0
                  </code>
                </div>
                <div className="text-[10px] text-charcoal-muted space-y-0.5">
                  <div>Scales representations:</div>
                  <div className="font-mono whitespace-nowrap text-center">
                    [res &times; &alpha;<sub>res</sub>, gru &times; &alpha;<sub>gru</sub>]
                  </div>
                </div>
              </div>
            </div>

            {/* Connector Arrow 2 */}
            <div className="flex items-center justify-center text-charcoal-muted/60 py-1 lg:py-0">
              <ArrowRight className="hidden lg:block w-5 h-5 text-sage" />
              <ArrowDown className="block lg:hidden w-5 h-5 text-sage" />
            </div>

            {/* Stage 3: 2-Token 8-Head Cross-Modal Attention */}
            <div className="flex-1 min-w-[220px] flex flex-col justify-center">
              <div className="text-[11px] font-semibold uppercase tracking-wider text-charcoal-muted text-center mb-3">
                Stage 3: Cross-Modal Attention
              </div>
              <div className="p-4 rounded-lg bg-sand border border-sandBorder space-y-2 text-center h-full flex flex-col justify-center">
                <div className="flex justify-center">
                  <div className="inline-flex p-2 rounded-full bg-white text-charcoal shadow-xs">
                    <Cpu className="w-4 h-4 text-charcoal" />
                  </div>
                </div>
                <h4 className="font-bold text-xs text-charcoal">
                  2-Token 8-Head Attention
                </h4>
                <p className="text-[11px] text-charcoal-muted leading-tight">
                  Tokenizes scaled representations into <span className="font-mono text-[10px] font-semibold">[Batch, 2, 512]</span>.
                </p>
                <div className="pt-1">
                  <span className="text-[10px] font-mono bg-white px-2 py-0.5 rounded border border-sandBorder inline-block text-charcoal">
                    Mean Pool &rarr; Residual FFN
                  </span>
                </div>
                <p className="text-[10px] text-charcoal-muted">
                  LayerNorm & Dropout (0.3) for cross-modal context refinement.
                </p>
              </div>
            </div>

            {/* Connector Arrow 3 */}
            <div className="flex items-center justify-center text-charcoal-muted/60 py-1 lg:py-0">
              <ArrowRight className="hidden lg:block w-5 h-5 text-sage" />
              <ArrowDown className="block lg:hidden w-5 h-5 text-sage" />
            </div>

            {/* Stage 4: Classification Head */}
            <div className="flex-1 min-w-[200px] flex flex-col justify-center">
              <div className="text-[11px] font-semibold uppercase tracking-wider text-charcoal-muted text-center mb-3">
                Stage 4: Prediction Head
              </div>
              <div className="p-4 rounded-lg bg-sand border border-sandBorder space-y-2 text-center h-full flex flex-col justify-center">
                <div className="flex justify-center">
                  <div className="inline-flex p-2 rounded-full bg-white text-terracotta shadow-xs">
                    <Sparkles className="w-4 h-4 text-terracotta" />
                  </div>
                </div>
                <h4 className="font-bold text-xs text-charcoal">
                  3-Stage MLP Classifier
                </h4>
                <p className="text-[11px] text-charcoal-muted leading-tight">
                  Dense layers (512 &rarr; 256 &rarr; 64 &rarr; 1) with GELU activations and dropout.
                </p>
                <div className="pt-1">
                  <div className="text-[10px] font-mono bg-white px-2 py-1 rounded border border-sandBorder inline-block text-charcoal font-bold">
                    Sigmoid &rarr; Risk Probability
                  </div>
                </div>
                <p className="text-[10px] text-charcoal-muted">
                  Compared against cutoff 0.41
                </p>
              </div>
            </div>

          </div>
        </div>
      </section>

      {/* Latent Input Embeddings Technical Section (Moved from Prediction Studio) */}
      <section className="space-y-4">
        <div className="flex items-center gap-2">
          <Database className="w-5 h-5 text-sage" />
          <h2 className="text-lg font-bold text-charcoal">
            Active Latent Input Embeddings & Precomputed Representations
          </h2>
        </div>

        <div className="bg-surface rounded-xl border border-sandBorder p-6 shadow-sm space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* ResNet34 Latent Vector Specs */}
            <div className="p-4 rounded-lg bg-sand/60 border border-sandBorder space-y-2">
              <div className="flex items-center justify-between">
                <span className="font-bold text-xs text-charcoal">ResNet34-1D Latent Vector</span>
                <span className="text-[10px] font-mono px-2 py-0.5 bg-white rounded border border-sandBorder font-semibold text-charcoal">
                  512 Dimensions
                </span>
              </div>
              <p className="text-xs text-charcoal-muted leading-relaxed">
                Represents static spatial morphology across the 12 ECG leads. Pre-extracted via a 34-layer 1-dimensional residual convolutional network trained to capture conduction delays, ST segment deviations, and QRS complex morphology.
              </p>
              <div className="flex items-center gap-2 pt-1 text-[11px] font-mono text-charcoal-muted">
                <span className="bg-white px-2 py-0.5 rounded border border-sandBorder">L2 Norm: Normalized</span>
                <span className="bg-white px-2 py-0.5 rounded border border-sandBorder">Domain: Waveform Morphology</span>
              </div>
            </div>

            {/* BiGRU Latent Vector Specs */}
            <div className="p-4 rounded-lg bg-sand/60 border border-sandBorder space-y-2">
              <div className="flex items-center justify-between">
                <span className="font-bold text-xs text-charcoal">BiGRU Latent Vector</span>
                <span className="text-[10px] font-mono px-2 py-0.5 bg-white rounded border border-sandBorder font-semibold text-charcoal">
                  256 Dimensions
                </span>
              </div>
              <p className="text-xs text-charcoal-muted leading-relaxed">
                Represents dynamic temporal rhythm and beat-to-beat cadence. Extracted via a 3-layer bidirectional recurrent gated unit with temporal attention pooling across sequential cardiac cycles.
              </p>
              <div className="flex items-center gap-2 pt-1 text-[11px] font-mono text-charcoal-muted">
                <span className="bg-white px-2 py-0.5 rounded border border-sandBorder">L2 Norm: Normalized</span>
                <span className="bg-white px-2 py-0.5 rounded border border-sandBorder">Domain: Temporal Cadence</span>
              </div>
            </div>
          </div>

          <div className="p-3.5 bg-sand/40 rounded-lg border border-sandBorder text-xs text-charcoal-muted flex items-start gap-2.5">
            <AlertTriangle className="w-4 h-4 text-terracotta shrink-0 mt-0.5" />
            <div className="leading-relaxed">
              <strong className="text-charcoal font-semibold">Upstream Feature Representation Notice:</strong> The deployed model performs inference directly on these high-dimensional latent vectors rather than raw analog microvolt ECG recordings. Raw 12-lead voltage waveforms were transformed upstream into these cached feature representations, guaranteeing high-throughput, low-latency cross-modal inference without local signal preprocessing overhead.
            </div>
          </div>
        </div>
      </section>

      {/* Interactive Plotly Analytics Section */}
      <section className="space-y-6">
        <h2 className="text-lg font-bold text-charcoal">
          Empirical Performance Visualizations
        </h2>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <ConfusionMatrixPlot />
          <MetricsComparisonPlot />
        </div>
      </section>

      {/* Verified Performance Benchmarks Table */}
      <section className="space-y-4">
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-bold text-charcoal">
            Verified Performance Metrics by Split & Threshold
          </h2>
          <span className="text-xs text-charcoal-muted">
            Model: Epoch 7 Peak Validation Checkpoint
          </span>
        </div>

        <div className="bg-surface rounded-xl border border-sandBorder overflow-hidden shadow-sm">
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead className="bg-sand border-b border-sandBorder text-charcoal font-semibold uppercase tracking-wider text-[11px]">
                <tr>
                  <th className="py-3 px-4">Evaluation Split</th>
                  <th className="py-3 px-4">Sample Size</th>
                  <th className="py-3 px-4">Threshold</th>
                  <th className="py-3 px-4">ROC-AUC</th>
                  <th className="py-3 px-4">Balanced Acc</th>
                  <th className="py-3 px-4">Recall (Sens)</th>
                  <th className="py-3 px-4">Specificity</th>
                  <th className="py-3 px-4">Precision</th>
                  <th className="py-3 px-4">F1-Score</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-sandBorder font-mono">
                {benchmarks && (
                  <>
                    <tr className="bg-sand/30 hover:bg-sand/50">
                      <td className="py-3 px-4 font-sans font-semibold text-charcoal">Validation (Optimal)</td>
                      <td className="py-3 px-4 font-medium text-charcoal-muted">N = {benchmarks.validation_metrics_at_041.sample_count}</td>
                      <td className="py-3 px-4 font-semibold text-charcoal">{benchmarks.validation_metrics_at_041.threshold}</td>
                      <td className="py-3 px-4 font-semibold text-charcoal">{benchmarks.validation_metrics_at_041.roc_auc}%</td>
                      <td className="py-3 px-4 font-semibold text-charcoal">{benchmarks.validation_metrics_at_041.balanced_accuracy}%</td>
                      <td className="py-3 px-4 font-semibold text-charcoal">{benchmarks.validation_metrics_at_041.recall_sensitivity}%</td>
                      <td className="py-3 px-4 font-semibold text-charcoal">{benchmarks.validation_metrics_at_041.specificity}%</td>
                      <td className="py-3 px-4 font-semibold text-charcoal">{benchmarks.validation_metrics_at_041.precision}%</td>
                      <td className="py-3 px-4 font-semibold text-charcoal">{benchmarks.validation_metrics_at_041.f1_score}%</td>
                    </tr>

                    <tr className="bg-white hover:bg-sand/20">
                      <td className="py-3 px-4 font-sans font-normal text-charcoal-muted">Validation (Standard)</td>
                      <td className="py-3 px-4 font-normal text-charcoal-muted">N = {benchmarks.validation_metrics_at_050.sample_count}</td>
                      <td className="py-3 px-4 font-normal text-charcoal-muted">{benchmarks.validation_metrics_at_050.threshold}</td>
                      <td className="py-3 px-4 font-normal text-charcoal-muted">{benchmarks.validation_metrics_at_050.roc_auc}%</td>
                      <td className="py-3 px-4 font-normal text-charcoal-muted">{benchmarks.validation_metrics_at_050.balanced_accuracy}%</td>
                      <td className="py-3 px-4 font-normal text-charcoal-muted">{benchmarks.validation_metrics_at_050.recall_sensitivity}%</td>
                      <td className="py-3 px-4 font-normal text-charcoal-muted">{benchmarks.validation_metrics_at_050.specificity}%</td>
                      <td className="py-3 px-4 font-normal text-charcoal-muted">{benchmarks.validation_metrics_at_050.precision}%</td>
                      <td className="py-3 px-4 font-normal text-charcoal-muted">{benchmarks.validation_metrics_at_050.f1_score}%</td>
                    </tr>

                    <tr className="bg-sand/30 hover:bg-sand/50">
                      <td className="py-3 px-4 font-sans font-semibold text-charcoal">Holdout Test (Evaluated)</td>
                      <td className="py-3 px-4 font-medium text-charcoal-muted">N = {benchmarks.test_metrics_at_041.sample_count}</td>
                      <td className="py-3 px-4 font-semibold text-charcoal">{benchmarks.test_metrics_at_041.threshold}</td>
                      <td className="py-3 px-4 font-semibold text-charcoal">{benchmarks.test_metrics_at_041.roc_auc}%</td>
                      <td className="py-3 px-4 font-semibold text-charcoal">{benchmarks.test_metrics_at_041.balanced_accuracy}%</td>
                      <td className="py-3 px-4 font-semibold text-charcoal">{benchmarks.test_metrics_at_041.recall_sensitivity}%</td>
                      <td className="py-3 px-4 font-semibold text-charcoal">{benchmarks.test_metrics_at_041.specificity}%</td>
                      <td className="py-3 px-4 font-semibold text-charcoal">{benchmarks.test_metrics_at_041.precision}%</td>
                      <td className="py-3 px-4 font-semibold text-charcoal">{benchmarks.test_metrics_at_041.f1_score}%</td>
                    </tr>

                    <tr className="bg-white hover:bg-sand/20">
                      <td className="py-3 px-4 font-sans font-normal text-charcoal-muted">Holdout Test (Standard)</td>
                      <td className="py-3 px-4 font-normal text-charcoal-muted">N = {benchmarks.test_metrics_at_050.sample_count}</td>
                      <td className="py-3 px-4 font-normal text-charcoal-muted">{benchmarks.test_metrics_at_050.threshold}</td>
                      <td className="py-3 px-4 font-normal text-charcoal-muted">{benchmarks.test_metrics_at_050.roc_auc}%</td>
                      <td className="py-3 px-4 font-normal text-charcoal-muted">{benchmarks.test_metrics_at_050.balanced_accuracy}%</td>
                      <td className="py-3 px-4 font-normal text-charcoal-muted">{benchmarks.test_metrics_at_050.recall_sensitivity}%</td>
                      <td className="py-3 px-4 font-normal text-charcoal-muted">{benchmarks.test_metrics_at_050.specificity}%</td>
                      <td className="py-3 px-4 font-normal text-charcoal-muted">{benchmarks.test_metrics_at_050.precision}%</td>
                      <td className="py-3 px-4 font-normal text-charcoal-muted">{benchmarks.test_metrics_at_050.f1_score}%</td>
                    </tr>
                  </>
                )}
              </tbody>
            </table>
          </div>
          <div className="p-3 bg-sand/40 border-t border-sandBorder text-[11px] text-charcoal-muted">
            Note: All metrics strictly reproduced from <code>best_fusion_model.zip</code> checkpoint. No metrics were assumed or cross-contaminated across splits.
          </div>
        </div>
      </section>
    </div>
  );
};

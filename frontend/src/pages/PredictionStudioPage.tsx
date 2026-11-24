import React, { useState, useEffect } from 'react';
import { AlertCircle, RefreshCw, Info, CheckCircle2 } from 'lucide-react';
import { fetchRecords, postPredictRawRecord } from '../api/client';
import { RecordsListResponse, RawPredictResponse } from '../types';
import { ECGWaveformPlot } from '../components/ECGWaveformPlot';
import { RiskResultCard } from '../components/RiskResultCard';
import { ReliabilityBars } from '../components/ReliabilityBars';

export const PredictionStudioPage: React.FC = () => {
  const [recordsData, setRecordsData] = useState<RecordsListResponse | null>(null);
  const [selectedRecordIndex, setSelectedRecordIndex] = useState<number>(42);
  const [isCustomIndex, setIsCustomIndex] = useState<boolean>(false);
  const [customInputValue, setCustomInputValue] = useState<string>('42');
  const [threshold] = useState<number>(0.41);

  const [prediction, setPrediction] = useState<RawPredictResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  // Fetch available validation records on mount
  useEffect(() => {
    let isMounted = true;
    async function loadRecords() {
      try {
        const data = await fetchRecords();
        if (isMounted) {
          setRecordsData(data);
        }
      } catch (err: unknown) {
        if (isMounted) {
          const msg = err instanceof Error ? err.message : 'Failed to load records';
          setError(msg);
        }
      }
    }
    loadRecords();
    return () => {
      isMounted = false;
    };
  }, []);

  // Run real end-to-end model inference on raw ECG waveform
  const handleAnalyze = async (recordIdx: number) => {
    setLoading(true);
    setError(null);
    try {
      const res = await postPredictRawRecord(recordIdx, threshold);
      setPrediction(res);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Raw ECG inference request failed';
      setError(msg);
    } finally {
      setLoading(false);
    }
  };

  // Run initial prediction on default sample 42 on load
  useEffect(() => {
    handleAnalyze(42);
  }, []);

  const handlePresetSelect = (idx: number) => {
    setIsCustomIndex(false);
    setSelectedRecordIndex(idx);
    setCustomInputValue(idx.toString());
    handleAnalyze(idx);
  };

  const handleCustomSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const parsed = parseInt(customInputValue, 10);
    if (isNaN(parsed) || parsed < 0 || parsed > (recordsData?.max_index ?? 4625)) {
      setError(`Please enter a valid record index between 0 and ${recordsData?.max_index ?? 4625}.`);
      return;
    }
    setSelectedRecordIndex(parsed);
    handleAnalyze(parsed);
  };

  return (
    <div className="space-y-8 py-6">
      {/* Page Header */}
      <div className="pb-4 border-b border-sandBorder">
        <h1 className="text-2xl sm:text-3xl font-extrabold text-charcoal tracking-tight">
          Prediction Studio
        </h1>
        <p className="text-xs sm:text-sm text-charcoal-muted mt-1">
          End-to-end inference on actual EchoNext 12-lead ECG waveforms using ResNet34-1D, BiGRU, and Adaptive Reliability Fusion.
        </p>
      </div>

      {/* Error Alert Banner */}
      {error && (
        <div className="p-4 rounded-xl bg-terracotta-light border border-terracotta-border flex items-start gap-3 text-sm text-terracotta">
          <AlertCircle className="w-5 h-5 flex-shrink-0 mt-0.5" />
          <div className="flex-1">
            <span className="font-semibold">Inference Error: </span>
            {error}
          </div>
          <button
            onClick={() => setError(null)}
            className="text-xs font-semibold underline hover:no-underline"
          >
            Dismiss
          </button>
        </div>
      )}

      {/* Record Selection Bar Card */}
      <div className="bg-surface rounded-xl border border-sandBorder p-5 shadow-sm space-y-4">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div className="space-y-1">
            <label className="text-xs font-semibold uppercase tracking-wider text-charcoal-muted">
              Select Patient Record ({recordsData ? `${recordsData.total_records} Cohort Records` : 'Loading...'})
            </label>
            <div className="flex flex-wrap items-center gap-2">
              {recordsData?.suggested_records.map((preset) => {
                const isSelected = !isCustomIndex && selectedRecordIndex === preset.index;
                return (
                  <button
                    key={preset.index}
                    onClick={() => handlePresetSelect(preset.index)}
                    className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
                      isSelected
                        ? 'bg-charcoal text-white shadow-sm'
                        : 'bg-sand hover:bg-sandBorder text-charcoal'
                    }`}
                  >
                    {preset.label}
                  </button>
                );
              })}

              <button
                onClick={() => setIsCustomIndex(true)}
                className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
                  isCustomIndex
                    ? 'bg-charcoal text-white shadow-sm'
                    : 'bg-sand hover:bg-sandBorder text-charcoal'
                }`}
              >
                Custom Index...
              </button>
            </div>
          </div>

          {/* Custom Index Form */}
          <div className="flex items-center gap-3">
            {isCustomIndex && (
              <form onSubmit={handleCustomSubmit} className="flex items-center gap-2">
                <input
                  type="number"
                  min={0}
                  max={recordsData?.max_index ?? 4625}
                  value={customInputValue}
                  onChange={(e) => setCustomInputValue(e.target.value)}
                  placeholder="0 - 4625"
                  className="w-24 px-3 py-1.5 text-xs bg-sand border border-sandBorder rounded-lg text-charcoal font-mono focus:outline-none focus:ring-1 focus:ring-sage"
                />
                <button
                  type="submit"
                  className="px-3 py-1.5 bg-sand hover:bg-sandBorder border border-sandBorder rounded-lg text-xs font-semibold text-charcoal"
                >
                  Load
                </button>
              </form>
            )}

            {loading && (
              <div className="inline-flex items-center gap-1.5 text-xs text-charcoal-muted">
                <RefreshCw className="w-3.5 h-3.5 animate-spin text-sage" />
                <span>Extracting features & inferring...</span>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Main Analysis Display Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
        {/* Left Column: ECG Waveform Preview (7 cols) */}
        <div className="lg:col-span-7 space-y-6">
          <ECGWaveformPlot
            recordIndex={selectedRecordIndex}
            waveform={prediction?.waveform ?? null}
            loading={loading}
          />
        </div>

        {/* Right Column: Prediction Result & Fusion Reliability (5 cols) */}
        <div className="lg:col-span-5 space-y-6">
          <RiskResultCard
            prediction={prediction}
            loading={loading}
            threshold={threshold}
          />

          <ReliabilityBars
            weights={prediction?.reliability_weights}
            embeddings={prediction?.embeddings_summary}
            loading={loading}
          />

          {/* Real Inference Provenance Card */}
          <div className="p-4 rounded-xl bg-surface border border-sandBorder shadow-2xs space-y-2">
            <div className="flex items-center gap-2 text-xs font-bold text-charcoal">
              <CheckCircle2 className="w-4 h-4 text-sage" />
              <span>Verified End-to-End Inference Pipeline</span>
            </div>
            <p className="text-[11px] text-charcoal-muted leading-relaxed">
              This prediction was generated directly from the 10-second 12-lead voltage waveform of Record #{selectedRecordIndex.toString().padStart(4, '0')}. The signal was processed by the trained 34-layer ResNet-1D (512-d) and 3-layer BiGRU (256-d) backbones before fusion classification.
            </p>
          </div>

          {/* Clinical Integrity Notice */}
          <div className="p-4 rounded-xl bg-sand/80 border border-sandBorder flex items-start gap-2.5 text-xs text-charcoal-muted">
            <Info className="w-4 h-4 text-sage flex-shrink-0 mt-0.5" />
            <div className="space-y-1">
              <span className="font-semibold text-charcoal">Investigational Prototype: </span>
              <span>
                CardioInsight is designed for research decision-support benchmarking. Risk probability scores and reliability gating should not be used for autonomous acute diagnostic decisions.
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

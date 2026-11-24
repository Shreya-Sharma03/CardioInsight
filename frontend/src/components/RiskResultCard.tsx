import React from 'react';
import { AlertTriangle, CheckCircle2 } from 'lucide-react';
import { PredictResponse, RawPredictResponse } from '../types';

interface RiskResultCardProps {
  prediction: PredictResponse | RawPredictResponse | null;
  loading: boolean;
  threshold: number;
  onThresholdChange?: (newThreshold: number) => void;
}

export const RiskResultCard: React.FC<RiskResultCardProps> = ({
  prediction,
  loading,
  threshold,
}) => {
  if (loading) {
    return (
      <div className="bg-surface rounded-xl border border-sandBorder p-6 shadow-sm animate-pulse">
        <div className="h-4 w-32 bg-sand rounded mb-4" />
        <div className="h-10 w-44 bg-sand rounded mb-3" />
        <div className="h-3 w-56 bg-sand rounded mb-6" />
        <div className="h-3 w-full bg-sand rounded" />
      </div>
    );
  }

  if (!prediction) {
    return (
      <div className="bg-surface rounded-xl border border-dashed border-sandBorder p-8 text-center text-charcoal-muted">
        <p className="text-sm">Select a validation record and click "Analyze Sample" to view real model predictions.</p>
      </div>
    );
  }

  const isHighRisk = prediction.is_high_risk;
  const probPct = prediction.probability * 100;
  const cutoffPct = threshold * 100;

  return (
    <div className="bg-surface rounded-xl border border-sandBorder p-6 shadow-sm">
      <div className="flex items-center justify-between mb-2">
        <span className="text-xs font-semibold uppercase tracking-wider text-charcoal-muted">
          Clinical Prediction
        </span>
        <span className="text-xs font-mono font-medium text-charcoal-muted">
          Record #{prediction.record_index}
        </span>
      </div>

      {/* Risk Badge & Icon */}
      <div className="flex items-center justify-between mt-2 mb-4">
        {isHighRisk ? (
          <div className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-md bg-terracotta-light border border-terracotta-border text-terracotta font-bold text-sm tracking-wide">
            <AlertTriangle className="w-4 h-4 text-terracotta" />
            <span>HIGH RISK</span>
          </div>
        ) : (
          <div className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-md bg-sage-light border border-sage-border text-sage-dark font-bold text-sm tracking-wide">
            <CheckCircle2 className="w-4 h-4 text-sage" />
            <span>LOW RISK</span>
          </div>
        )}
      </div>

      {/* Large Probability Display */}
      <div className="my-3">
        <div
          className={`text-5xl font-extrabold tracking-tight ${
            isHighRisk ? 'text-terracotta' : 'text-sage-dark'
          }`}
        >
          {probPct.toFixed(1)}%
        </div>
        <p className="text-xs text-charcoal-muted mt-1">
          Calculated heart failure risk probability derived from multimodal fusion
        </p>
      </div>

      {/* Threshold Visual Progress Gauge */}
      <div className="mt-5 pt-3">
        <div className="relative h-2.5 w-full bg-sand rounded-full overflow-hidden">
          {/* Probability fill */}
          <div
            className={`h-full rounded-full transition-all duration-500 ${
              isHighRisk ? 'bg-terracotta' : 'bg-sage'
            }`}
            style={{ width: `${Math.min(100, Math.max(0, probPct))}%` }}
          />
        </div>

        {/* Cutoff tick indicator */}
        <div className="relative w-full text-xs text-charcoal-subtle mt-1.5 flex justify-between font-mono text-[10px]">
          <span>0%</span>
          <span
            className="absolute -translate-x-1/2 text-charcoal font-semibold"
            style={{ left: `${cutoffPct}%` }}
          >
            ▲ Cutoff (41%)
          </span>
          <span>100%</span>
        </div>
      </div>
    </div>
  );
};

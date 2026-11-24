import React from 'react';
import { Layers } from 'lucide-react';
import { ReliabilityWeights, EmbeddingsSummary } from '../types';

interface ReliabilityBarsProps {
  weights?: ReliabilityWeights;
  embeddings?: EmbeddingsSummary;
  loading?: boolean;
}

export const ReliabilityBars: React.FC<ReliabilityBarsProps> = ({
  weights,
  embeddings,
  loading,
}) => {
  if (loading) {
    return (
      <div className="bg-surface rounded-xl border border-sandBorder p-6 shadow-sm animate-pulse space-y-4">
        <div className="h-4 w-40 bg-sand rounded" />
        <div className="h-8 w-full bg-sand rounded" />
        <div className="h-8 w-full bg-sand rounded" />
      </div>
    );
  }

  if (!weights) {
    return null;
  }

  const resnetPct = weights.resnet34 * 100;
  const bigruPct = weights.bigru * 100;

  return (
    <div className="bg-surface rounded-xl border border-sandBorder p-6 shadow-sm">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <Layers className="w-4 h-4 text-sage" />
          <h3 className="text-sm font-semibold text-charcoal uppercase tracking-wider text-xs">
            Adaptive Reliability Fusion Weights
          </h3>
        </div>
        <span className="text-[11px] font-mono text-charcoal-muted">
          Sum: {(resnetPct + bigruPct).toFixed(1)}%
        </span>
      </div>

      <div className="space-y-4">
        {/* ResNet34-1D Bar */}
        <div>
          <div className="flex justify-between items-center text-xs mb-1.5">
            <span className="font-medium text-charcoal">
              ResNet34-1D (Morphological Branch)
            </span>
            <span className="font-mono font-semibold text-charcoal">
              {resnetPct.toFixed(1)}%
            </span>
          </div>
          <div className="h-2.5 w-full bg-sand rounded-full overflow-hidden">
            <div
              className="h-full bg-sage rounded-full transition-all duration-500"
              style={{ width: `${resnetPct}%` }}
            />
          </div>
          {embeddings && (
            <p className="text-[11px] text-charcoal-muted mt-1">
              {embeddings.resnet_dim} dims · Norm: {embeddings.resnet_norm}
            </p>
          )}
        </div>

        {/* BiGRU Bar */}
        <div>
          <div className="flex justify-between items-center text-xs mb-1.5">
            <span className="font-medium text-charcoal">
              BiGRU (Temporal Dynamics Branch)
            </span>
            <span className="font-mono font-semibold text-charcoal">
              {bigruPct.toFixed(1)}%
            </span>
          </div>
          <div className="h-2.5 w-full bg-sand rounded-full overflow-hidden">
            <div
              className="h-full bg-[#6C7A89] rounded-full transition-all duration-500"
              style={{ width: `${bigruPct}%` }}
            />
          </div>
          {embeddings && (
            <p className="text-[11px] text-charcoal-muted mt-1">
              {embeddings.bigru_dim} dims · Norm: {embeddings.bigru_norm}
            </p>
          )}
        </div>
      </div>

      <p className="text-[11px] text-charcoal-muted mt-4 pt-3 border-t border-sandBorder">
        Dynamic softmax gating assigns per-sample scalar weights (&alpha;<sub>resnet</sub>, &alpha;<sub>bigru</sub>) based on embedding confidence before cross-modal self-attention.
      </p>
    </div>
  );
};

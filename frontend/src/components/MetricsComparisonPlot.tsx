import React, { useEffect, useRef } from 'react';
import Plotly from 'plotly.js-dist-min';

export const MetricsComparisonPlot: React.FC = () => {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!containerRef.current) return;

    const categories = ['ROC-AUC', 'Balanced Acc', 'Sensitivity', 'Specificity', 'F1-Score'];

    // Verified metrics at threshold = 0.41 (Epoch 7 Best Checkpoint: audit_results/reproduced_evaluation_report.md)
    const valMetrics = [89.12, 82.33, 79.21, 85.45, 65.36];
    const testMetrics = [88.44, 81.01, 76.51, 85.51, 62.72];

    const data: Plotly.Data[] = [
      {
        x: categories,
        y: valMetrics,
        name: 'Validation Split (N=4,626)',
        type: 'bar',
        marker: {
          color: '#4A7C59', // Sage green
        },
        hovertemplate: 'Validation %{x}: %{y:.2f}%<extra></extra>',
      },
      {
        x: categories,
        y: testMetrics,
        name: 'Holdout Test Split (N=5,442)',
        type: 'bar',
        marker: {
          color: '#6C7A89', // Slate
        },
        hovertemplate: 'Test %{x}: %{y:.2f}%<extra></extra>',
      },
    ];

    const layout: Partial<Plotly.Layout> = {
      autosize: true,
      height: 290,
      barmode: 'group',
      margin: { l: 45, r: 20, t: 30, b: 40 },
      paper_bgcolor: '#FFFFFF',
      plot_bgcolor: '#FCFAF7',
      legend: {
        orientation: 'h',
        x: 0.1,
        y: 1.15,
        font: { size: 11, color: '#1A1C1E' },
      },
      yaxis: {
        title: { text: 'Percentage (%)', font: { size: 10, color: '#5A6065' } },
        range: [50, 95],
        tickfont: { size: 10, color: '#5A6065' },
        gridcolor: '#E8DED6',
      },
      xaxis: {
        tickfont: { size: 11, color: '#1A1C1E' },
      },
    };

    const config: Partial<Plotly.Config> = {
      responsive: true,
      displayModeBar: false,
    };

    Plotly.newPlot(containerRef.current, data, layout, config);

    const handleResize = () => {
      if (containerRef.current) {
        Plotly.Plots.resize(containerRef.current);
      }
    };
    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      if (containerRef.current) {
        Plotly.purge(containerRef.current);
      }
    };
  }, []);

  return (
    <div className="bg-surface rounded-xl border border-sandBorder p-5 shadow-sm">
      <div className="flex items-center justify-between mb-2">
        <h3 className="text-sm font-semibold text-charcoal uppercase tracking-wider text-xs">
          Validation vs. Test Generalization
        </h3>
        <span className="text-[11px] font-mono text-charcoal-muted">
          Decision Threshold: 0.41 · Peak Epoch 7
        </span>
      </div>
      <div ref={containerRef} className="w-full rounded-lg overflow-hidden border border-sandBorder" />
    </div>
  );
};

import React, { useEffect, useRef } from 'react';
import Plotly from 'plotly.js-dist-min';

export const ConfusionMatrixPlot: React.FC = () => {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!containerRef.current) return;

    // Exact verified test set confusion matrix (Epoch 7 Best Checkpoint, Threshold = 0.41, N = 5,442)
    // Row 0: Actual Low Risk (0) -> TN = 3831, FP = 649
    // Row 1: Actual High Risk (1) -> FN = 226, TP = 736
    // We map distinct numerical codes to each cell to render 4 unique, accessible categorical colors:
    // TN (0) -> Soft muted sage green (#D1E7DD)
    // FP (1) -> Soft amber/warning yellow (#FFF3CD)
    // FN (2) -> Soft coral/terracotta (#F8D7DA)
    // TP (3) -> Deeper teal-sage (#A3CFBB)
    const zCategories = [
      [0, 1], // TN, FP
      [2, 3], // FN, TP
    ];

    const xLabels = ['Pred Low Risk (0)', 'Pred High Risk (1)'];
    const yLabels = ['Actual Low Risk (0)', 'Actual High Risk (1)'];

    const annotations: Partial<Plotly.Annotations>[] = [
      {
        x: 'Pred Low Risk (0)',
        y: 'Actual Low Risk (0)',
        text: '<b style="font-size:16px;color:#0F5132">3,831</b><br><span style="font-size:11px;font-weight:600;color:#0F5132">True Negative (85.5%)</span><br><span style="font-size:10px;color:#146C43">Correct Non-HF</span>',
        showarrow: false,
      },
      {
        x: 'Pred High Risk (1)',
        y: 'Actual Low Risk (0)',
        text: '<b style="font-size:16px;color:#664D03">649</b><br><span style="font-size:11px;font-weight:600;color:#664D03">False Positive (14.5%)</span><br><span style="font-size:10px;color:#664D03">Type I Error</span>',
        showarrow: false,
      },
      {
        x: 'Pred Low Risk (0)',
        y: 'Actual High Risk (1)',
        text: '<b style="font-size:16px;color:#842029">226</b><br><span style="font-size:11px;font-weight:600;color:#842029">False Negative (23.5%)</span><br><span style="font-size:10px;color:#842029">Type II Error (Missed)</span>',
        showarrow: false,
      },
      {
        x: 'Pred High Risk (1)',
        y: 'Actual High Risk (1)',
        text: '<b style="font-size:16px;color:#084298">736</b><br><span style="font-size:11px;font-weight:600;color:#084298">True Positive (76.5%)</span><br><span style="font-size:10px;color:#052C65">Correct Flagged HF</span>',
        showarrow: false,
      },
    ];

    const data: Plotly.Data[] = [
      {
        z: zCategories,
        x: xLabels,
        y: yLabels,
        type: 'heatmap',
        colorscale: [
          [0.0, '#D1E7DD'],   // TN: Muted Mint Green
          [0.33, '#FFF3CD'],  // FP: Muted Warm Amber
          [0.66, '#F8D7DA'],  // FN: Muted Rose / Terracotta
          [1.0, '#CFE2FF'],   // TP: Muted Soft Blue
        ],
        zmin: 0,
        zmax: 3,
        showscale: false,
        hoverinfo: 'none',
      },
    ];

    const layout: Partial<Plotly.Layout> = {
      autosize: true,
      height: 290,
      margin: { l: 140, r: 25, t: 20, b: 50 },
      paper_bgcolor: '#FFFFFF',
      plot_bgcolor: '#FFFFFF',
      dragmode: false,
      xaxis: {
        tickfont: { size: 12, color: '#1A1C1E', family: 'sans-serif' },
        side: 'bottom',
        fixedrange: true,
      },
      yaxis: {
        tickfont: { size: 12, color: '#1A1C1E', family: 'sans-serif' },
        autorange: 'reversed',
        fixedrange: true,
      },
      annotations: annotations,
    };

    const config: Partial<Plotly.Config> = {
      responsive: true,
      displayModeBar: false,
      staticPlot: true,
      doubleClick: false,
      scrollZoom: false,
      showAxisDragHandles: false,
      showAxisRangeEntryBoxes: false,
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
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-semibold text-charcoal uppercase tracking-wider text-xs">
          Verified Test-Set Confusion Matrix
        </h3>
        <span className="text-[11px] font-mono text-charcoal-muted">
          Holdout Test Set (N = 5,442) · Threshold: 0.41
        </span>
      </div>

      <div ref={containerRef} className="w-full rounded-lg overflow-hidden border border-sandBorder" />

      {/* Visual Category Legend */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 mt-3 pt-3 border-t border-sandBorder text-[11px]">
        <div className="flex items-center gap-1.5">
          <span className="w-3 h-3 rounded bg-[#D1E7DD] border border-[#0F5132]/30 flex-shrink-0" />
          <span className="text-charcoal"><b>TN:</b> True Negative (3,831)</span>
        </div>
        <div className="flex items-center gap-1.5">
          <span className="w-3 h-3 rounded bg-[#FFF3CD] border border-[#664D03]/30 flex-shrink-0" />
          <span className="text-charcoal"><b>FP:</b> False Positive (649)</span>
        </div>
        <div className="flex items-center gap-1.5">
          <span className="w-3 h-3 rounded bg-[#F8D7DA] border border-[#842029]/30 flex-shrink-0" />
          <span className="text-charcoal"><b>FN:</b> False Negative (226)</span>
        </div>
        <div className="flex items-center gap-1.5">
          <span className="w-3 h-3 rounded bg-[#CFE2FF] border border-[#084298]/30 flex-shrink-0" />
          <span className="text-charcoal"><b>TP:</b> True Positive (736)</span>
        </div>
      </div>

      <div className="mt-2.5 flex flex-wrap justify-between text-xs text-charcoal-muted px-1">
        <span>Sensitivity (Recall): <b className="text-charcoal">76.51%</b></span>
        <span>Specificity: <b className="text-charcoal">85.51%</b></span>
        <span>Balanced Accuracy: <b className="text-charcoal">81.01%</b></span>
      </div>
    </div>
  );
};

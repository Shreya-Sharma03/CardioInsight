import React, { useEffect, useRef, useState } from 'react';
import Plotly from 'plotly.js-dist-min';
import { Activity, Grid, Maximize2, LayoutGrid } from 'lucide-react';
import { WaveformData } from '../types';

interface ECGWaveformPlotProps {
  recordIndex: number;
  waveform: WaveformData | null;
  loading?: boolean;
}

const LEAD_DESCRIPTIONS: Record<string, string> = {
  I: 'Lead I (Lateral)',
  II: 'Lead II (Inferior / Rhythm)',
  III: 'Lead III (Inferior)',
  aVR: 'Lead aVR (Right-sided)',
  aVL: 'Lead aVL (High Lateral)',
  aVF: 'Lead aVF (Inferior)',
  V1: 'Lead V1 (Septal)',
  V2: 'Lead V2 (Septal)',
  V3: 'Lead V3 (Anterior)',
  V4: 'Lead V4 (Anterior)',
  V5: 'Lead V5 (Lateral)',
  V6: 'Lead V6 (Lateral)',
};

export const ECGWaveformPlot: React.FC<ECGWaveformPlotProps> = ({
  recordIndex,
  waveform,
  loading = false,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const [viewMode, setViewMode] = useState<'all' | 'lead2'>('all');

  useEffect(() => {
    if (!containerRef.current || !waveform) return;

    const leadNames = waveform.lead_names;
    const timeAxis = waveform.time_axis;

    if (viewMode === 'lead2') {
      // Single Lead II Detailed Rhythm View
      const leadSignal = waveform.leads['II'] || waveform.leads[leadNames[0]] || [];
      const trace: Plotly.Data = {
        x: timeAxis,
        y: leadSignal,
        type: 'scatter',
        mode: 'lines',
        name: 'Lead II',
        line: { color: '#C05638', width: 1.5 },
        hoverinfo: 'x+y',
        hovertemplate: '<b>Lead II</b><br>Time: %{x:.2f}s<br>Amplitude: %{y:.3f}<extra></extra>',
      };

      const layout: Partial<Plotly.Layout> = {
        autosize: true,
        height: 320,
        margin: { l: 45, r: 20, t: 30, b: 40 },
        plot_bgcolor: '#FCFAF7',
        paper_bgcolor: '#FFFFFF',
        showlegend: false,
        xaxis: {
          title: { text: 'Time (seconds)', font: { size: 11, color: '#5A6065' } },
          range: [0, 10],
          dtick: 1.0,
          gridcolor: '#F0E6DD',
          gridwidth: 1,
          zeroline: false,
        },
        yaxis: {
          title: { text: 'Normalized Amplitude', font: { size: 11, color: '#5A6065' } },
          gridcolor: '#F0E6DD',
          gridwidth: 1,
          zeroline: true,
          zerolinecolor: '#DFD5CC',
        },
      };

      const config: Partial<Plotly.Config> = {
        responsive: true,
        displayModeBar: false,
        staticPlot: false,
      };

      Plotly.newPlot(containerRef.current, [trace], layout, config);
    } else {
      // Complete 12-Lead Multi-Subplot Grid (6 rows x 2 cols or 12 stacked)
      // Standard clinical layout: 6 rows, 2 columns (Limb leads col 1, Chest leads col 2)
      // Col 1: I, II, III, aVR, aVL, aVF
      // Col 2: V1, V2, V3, V4, V5, V6
      const numRows = 6;
      const numCols = 2;
      const traces: Plotly.Data[] = [];
      const annotations: Partial<Plotly.Annotations>[] = [];

      const col1Leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF'];
      const col2Leads = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6'];

      // Build 12 subplots
      for (let r = 0; r < numRows; r++) {
        for (let c = 0; c < numCols; c++) {
          const lead = c === 0 ? col1Leads[r] : col2Leads[r];
          const leadSignal = waveform.leads[lead] || [];
          const plotIndex = r * numCols + c + 1;
          const xAxisKey = plotIndex === 1 ? 'x' : `x${plotIndex}`;
          const yAxisKey = plotIndex === 1 ? 'y' : `y${plotIndex}`;

          traces.push({
            x: timeAxis,
            y: leadSignal,
            xaxis: xAxisKey,
            yaxis: yAxisKey,
            type: 'scatter',
            mode: 'lines',
            line: {
              color: c === 0 ? '#1A1C1E' : '#2A4365',
              width: 1.2,
            },
            name: lead,
            hoverinfo: 'x+y',
            hovertemplate: `<b>${lead}</b>: %{y:.2f} (t=%{x:.2f}s)<extra></extra>`,
          });

          // Label annotation for each lead subplot
          annotations.push({
            xref: (plotIndex === 1 ? 'x domain' : `x${plotIndex} domain`) as any,
            yref: (plotIndex === 1 ? 'y domain' : `y${plotIndex} domain`) as any,
            x: 0.02,
            y: 0.90,
            text: `<b>${lead}</b>`,
            showarrow: false,
            font: { size: 11, color: '#1A1C1E', family: 'monospace' },
            bgcolor: 'rgba(255, 255, 255, 0.85)',
            bordercolor: '#E5E0D8',
            borderwidth: 1,
            borderpad: 2,
          });
        }
      }

      // Generate layout with subplots grid
      const gridLayout: Record<string, any> = {
        autosize: true,
        height: 680,
        margin: { l: 40, r: 25, t: 25, b: 35 },
        plot_bgcolor: '#FCFAF7',
        paper_bgcolor: '#FFFFFF',
        showlegend: false,
        grid: {
          rows: numRows,
          columns: numCols,
          pattern: 'independent',
          roworder: 'top to bottom',
        },
        annotations: annotations,
      };

      // Set axes configs
      for (let p = 1; p <= 12; p++) {
        const xKey = p === 1 ? 'xaxis' : `xaxis${p}`;
        const yKey = p === 1 ? 'yaxis' : `yaxis${p}`;
        const isBottom = p === 11 || p === 12;

        gridLayout[xKey] = {
          range: [0, 10],
          showticklabels: isBottom,
          dtick: 2.0,
          gridcolor: '#F2EBE4',
          gridwidth: 0.8,
          zeroline: false,
          tickfont: { size: 9, color: '#666' },
        };

        gridLayout[yKey] = {
          gridcolor: '#F2EBE4',
          gridwidth: 0.8,
          zeroline: true,
          zerolinecolor: '#E0D5CB',
          tickfont: { size: 8, color: '#666' },
          showticklabels: false,
        };
      }

      const config: Partial<Plotly.Config> = {
        responsive: true,
        displayModeBar: false,
      };

      Plotly.newPlot(containerRef.current, traces, gridLayout, config);
    }

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
  }, [waveform, viewMode]);

  return (
    <div className="bg-surface rounded-xl border border-sandBorder p-5 shadow-sm space-y-3">
      {/* Header with lead mode switcher and record badge */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-2 border-b border-sandBorder">
        <div className="flex items-center gap-2">
          <Activity className="w-5 h-5 text-terracotta" />
          <div>
            <h3 className="text-sm font-bold text-charcoal flex items-center gap-2">
              Actual EchoNext 12-Lead ECG Waveform
              <span className="text-xs font-mono font-normal bg-sand px-2 py-0.5 rounded border border-sandBorder text-charcoal">
                Record #{recordIndex.toString().padStart(4, '0')}
              </span>
            </h3>
            <p className="text-[11px] text-charcoal-muted">
              10.0s recording · 250 Hz · 12 simultaneous physiological leads · Preprocessed input to ResNet34 & BiGRU
            </p>
          </div>
        </div>

        {/* View mode toggle */}
        <div className="flex items-center gap-1.5 self-start sm:self-auto bg-sand p-1 rounded-lg border border-sandBorder">
          <button
            type="button"
            onClick={() => setViewMode('all')}
            className={`flex items-center gap-1 px-2.5 py-1 text-xs font-medium rounded transition-colors ${
              viewMode === 'all'
                ? 'bg-white text-charcoal shadow-2xs font-semibold'
                : 'text-charcoal-muted hover:text-charcoal'
            }`}
          >
            <LayoutGrid className="w-3.5 h-3.5" />
            12-Lead Grid
          </button>
          <button
            type="button"
            onClick={() => setViewMode('lead2')}
            className={`flex items-center gap-1 px-2.5 py-1 text-xs font-medium rounded transition-colors ${
              viewMode === 'lead2'
                ? 'bg-white text-charcoal shadow-2xs font-semibold'
                : 'text-charcoal-muted hover:text-charcoal'
            }`}
          >
            <Maximize2 className="w-3.5 h-3.5" />
            Lead II Rhythm
          </button>
        </div>
      </div>

      {/* Waveform Visualization Canvas */}
      <div className="relative">
        {loading && (
          <div className="absolute inset-0 bg-white/70 backdrop-blur-2xs flex items-center justify-center z-10 rounded-lg">
            <div className="flex items-center gap-2 text-xs font-semibold text-charcoal bg-white px-3 py-2 rounded-lg border border-sandBorder shadow-sm">
              <Activity className="w-4 h-4 text-terracotta animate-pulse" />
              Loading actual ECG waveform #{recordIndex}...
            </div>
          </div>
        )}

        <div
          ref={containerRef}
          className="w-full rounded-lg overflow-hidden border border-sandBorder bg-[#FCFAF7]"
        />
      </div>

      {/* Grid Legend & Lead Mapping */}
      <div className="grid grid-cols-2 sm:grid-cols-4 md:grid-cols-6 gap-2 pt-2 text-[11px] text-charcoal-muted border-t border-sandBorder font-mono">
        <div><strong className="text-charcoal font-sans">I:</strong> Lateral</div>
        <div><strong className="text-charcoal font-sans">II:</strong> Inferior / Rhythm</div>
        <div><strong className="text-charcoal font-sans">III:</strong> Inferior</div>
        <div><strong className="text-charcoal font-sans">aVR:</strong> Right-sided</div>
        <div><strong className="text-charcoal font-sans">aVL:</strong> High Lateral</div>
        <div><strong className="text-charcoal font-sans">aVF:</strong> Inferior</div>
        <div><strong className="text-charcoal font-sans">V1:</strong> Septal</div>
        <div><strong className="text-charcoal font-sans">V2:</strong> Septal</div>
        <div><strong className="text-charcoal font-sans">V3:</strong> Anterior</div>
        <div><strong className="text-charcoal font-sans">V4:</strong> Anterior</div>
        <div><strong className="text-charcoal font-sans">V5:</strong> Lateral</div>
        <div><strong className="text-charcoal font-sans">V6:</strong> Lateral</div>
      </div>
    </div>
  );
};

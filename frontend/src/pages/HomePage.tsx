import React from 'react';
import { ArrowRight, Activity, BookOpen, ChevronRight } from 'lucide-react';
import { ActivePage } from '../types';

interface HomePageProps {
  onNavigate: (page: ActivePage) => void;
}

export const HomePage: React.FC<HomePageProps> = ({ onNavigate }) => {
  return (
    <div className="space-y-16 sm:space-y-20 py-8 md:py-14 max-w-5xl mx-auto">
      {/* Hero Section */}
      <section className="relative overflow-hidden rounded-2xl bg-[#F2EFE7] border-2 border-[#D8D0BE] p-8 sm:p-12 md:p-16 shadow-sm">
        {/* Subtle decorative ECG-inspired visual detail in muted rust (#B54E35) at ~38% opacity, stroke 2px */}
        <div className="absolute -right-4 sm:right-0 top-1/2 -translate-y-1/2 w-80 sm:w-96 h-48 opacity-[0.38] pointer-events-none select-none overflow-hidden">
          <svg
            className="w-full h-full text-[#B54E35]"
            viewBox="0 0 600 200"
            fill="none"
            xmlns="http://www.w3.org/2000/svg"
            aria-hidden="true"
          >
            <path
              d="M0,100 L160,100 L180,75 L195,125 L215,15 L235,175 L250,85 L265,100 L320,100 L340,90 L360,100 L600,100"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
        </div>

        <div className="relative max-w-2xl space-y-6">
          {/* Subtle Project Tag with rust dot */}
          <div className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-[#FFFFFF] border border-[#D8D0BE] text-[11px] font-bold text-[#343D35] uppercase tracking-wider shadow-2xs">
            <span className="w-2 h-2 rounded-full bg-[#B54E35]" />
            CardioInsight Research
          </div>

          {/* Main Headline */}
          <h1 className="text-3xl sm:text-4xl md:text-5xl font-extrabold tracking-tight text-[#343D35] leading-[1.18]">
            Understanding Heart Function Through ECG Signals
          </h1>

          {/* Supporting Description - max-w-xl prevents collision with ECG baseline on right */}
          <p className="text-base sm:text-lg text-[#343D35] leading-relaxed font-normal max-w-xl">
            CardioInsight is a research prototype that uses deep learning to identify ECG patterns associated with reduced heart pumping function.
          </p>

          {/* Prominent CTAs - Both consistent dark charcoal (#2C3531) with hover (#222926) */}
          <div className="pt-2 flex flex-wrap items-center gap-4">
            <button
              type="button"
              onClick={() => onNavigate('prediction')}
              className="inline-flex items-center gap-2.5 px-6 py-3 rounded-lg bg-[#2C3531] hover:bg-[#222926] text-[#FFFFFF] font-semibold text-sm transition-all shadow-sm hover:shadow"
            >
              <span>Explore ECG Analysis</span>
              <ArrowRight className="w-4 h-4 text-[#FFFFFF]" />
            </button>

            <button
              type="button"
              onClick={() => onNavigate('insights')}
              className="inline-flex items-center gap-2 px-5 py-3 rounded-lg bg-[#2C3531] hover:bg-[#222926] text-[#FFFFFF] font-semibold text-sm transition-all shadow-sm hover:shadow"
            >
              <span>About the Research</span>
              <ChevronRight className="w-4 h-4 text-[#FFFFFF]" />
            </button>
          </div>
        </div>

        {/* Minimalist ECG divider line at base of hero with rust dash */}
        <div className="mt-12 pt-6 border-t-2 border-[#D8D0BE] flex items-center justify-between text-xs text-[#343D35] font-mono">
          <div className="flex items-center gap-2">
            <span className="w-3 h-1 rounded-full bg-[#B54E35]" />
            <span className="font-bold text-[#343D35]">EchoNext Validation Cohort</span>
          </div>
          <div className="hidden sm:block tracking-wider font-semibold text-[#343D35]">
            12-Lead Electrocardiography
          </div>
        </div>
      </section>

      {/* Explore CardioInsight Section (Two Main Destinations) */}
      <section className="space-y-6">
        <div className="space-y-1">
          <h2 className="text-xl sm:text-2xl font-bold tracking-tight text-[#343D35]">
            Explore CardioInsight
          </h2>
          <p className="text-xs sm:text-sm text-[#343D35] font-medium opacity-90">
            Navigate the interactive analysis workspace or review the academic methodology.
          </p>
        </div>

        <div className="grid md:grid-cols-2 gap-6">
          {/* Card 1: ECG Analysis */}
          <div
            onClick={() => onNavigate('prediction')}
            className="group cursor-pointer rounded-xl bg-[#FFFFFF] border-2 border-[#D8D0BE] p-7 shadow-2xs hover:border-[#89977D] hover:shadow-md transition-all flex flex-col justify-between"
          >
            <div className="space-y-4">
              <div className="w-11 h-11 rounded-lg bg-[#F2EFE7] border border-[#D8D0BE] flex items-center justify-center text-[#89977D] group-hover:bg-[#89977D] group-hover:text-[#FFFFFF] group-hover:border-[#89977D] transition-colors shadow-2xs">
                <Activity className="w-5 h-5 stroke-[2.2]" />
              </div>
              <div className="space-y-1.5">
                <h3 className="text-lg font-bold text-[#343D35] group-hover:text-[#89977D] transition-colors">
                  ECG Analysis
                </h3>
                <p className="text-sm text-[#343D35] leading-relaxed opacity-90">
                  Explore ECG records, examine their waveforms and view corresponding risk predictions.
                </p>
              </div>
            </div>

            {/* Bottom action link with rust hover (#B54E35) */}
            <div className="pt-6 mt-6 border-t border-[#D8D0BE] flex items-center justify-between text-xs font-bold text-[#89977D] group-hover:text-[#B54E35] transition-colors">
              <span>Explore ECG Analysis</span>
              <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
            </div>
          </div>

          {/* Card 2: Research Overview */}
          <div
            onClick={() => onNavigate('insights')}
            className="group cursor-pointer rounded-xl bg-[#FFFFFF] border-2 border-[#D8D0BE] p-7 shadow-2xs hover:border-[#B9785D] hover:shadow-md transition-all flex flex-col justify-between"
          >
            <div className="space-y-4">
              <div className="w-11 h-11 rounded-lg bg-[#F2EFE7] border border-[#D8D0BE] flex items-center justify-center text-[#B9785D] group-hover:bg-[#B9785D] group-hover:text-[#FFFFFF] group-hover:border-[#B9785D] transition-colors shadow-2xs">
                <BookOpen className="w-5 h-5 stroke-[2.2]" />
              </div>
              <div className="space-y-1.5">
                <h3 className="text-lg font-bold text-[#343D35] group-hover:text-[#B9785D] transition-colors">
                  Research Overview
                </h3>
                <p className="text-sm text-[#343D35] leading-relaxed opacity-90">
                  Learn about the research objective, dataset and evaluation approach behind CardioInsight.
                </p>
              </div>
            </div>

            {/* Bottom action link with rust hover (#B54E35) */}
            <div className="pt-6 mt-6 border-t border-[#D8D0BE] flex items-center justify-between text-xs font-bold text-[#B9785D] group-hover:text-[#B54E35] transition-colors">
              <span>Explore Model Insights</span>
              <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
            </div>
          </div>
        </div>
      </section>

      {/* Research Disclaimer Section */}
      <section className="rounded-xl bg-[#F2EFE7] border-2 border-[#D8D0BE] p-6 sm:p-7 shadow-2xs">
        <div className="max-w-3xl">
          <div className="text-[11px] font-bold uppercase tracking-wider text-[#343D35]/80 mb-1.5">
            Research Disclaimer
          </div>
          <p className="text-xs sm:text-sm text-[#343D35] leading-relaxed font-normal">
            CardioInsight is a research prototype, not a diagnostic tool. Its predictions should not replace professional medical advice or clinical evaluation.
          </p>
        </div>
      </section>
    </div>
  );
};

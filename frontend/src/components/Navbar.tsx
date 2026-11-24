import React from 'react';
import { Activity } from 'lucide-react';
import { ActivePage, HealthResponse } from '../types';

interface NavbarProps {
  activePage: ActivePage;
  onNavigate: (page: ActivePage) => void;
  health: HealthResponse | null;
  healthLoading: boolean;
}

export const Navbar: React.FC<NavbarProps> = ({
  activePage,
  onNavigate,
  health,
  healthLoading,
}) => {
  const navItems: { id: ActivePage; label: string }[] = [
    { id: 'home', label: 'Home' },
    { id: 'prediction', label: 'Prediction Studio' },
    { id: 'insights', label: 'Model Insights' },
  ];

  return (
    <header className="sticky top-0 z-50 bg-[#FBF9F5]/90 backdrop-blur-md border-b border-sandBorder">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
        {/* Brand / Logo */}
        <div
          onClick={() => onNavigate('home')}
          className="flex items-center gap-3 cursor-pointer group"
        >
          <div className="w-9 h-9 rounded-lg bg-sage text-white flex items-center justify-center shadow-sm group-hover:bg-sage-dark transition-colors">
            <Activity className="w-5 h-5 animate-pulse" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="font-bold text-lg tracking-tight text-charcoal">
                CardioInsight
              </span>
              <span className="text-[10px] font-semibold uppercase tracking-wider px-2 py-0.5 rounded-full bg-sand text-charcoal-muted border border-sandBorder">
                Research v1.0
              </span>
            </div>
            <p className="text-xs text-charcoal-muted hidden sm:block">
              ECG-based Heart Failure Risk Analysis
            </p>
          </div>
        </div>

        {/* Navigation Tabs */}
        <nav className="flex items-center gap-1 sm:gap-2">
          {navItems.map((item) => {
            const isActive = activePage === item.id;
            return (
              <button
                key={item.id}
                onClick={() => onNavigate(item.id)}
                className={`px-3.5 py-1.5 rounded-md text-sm font-medium transition-all ${
                  isActive
                    ? 'bg-charcoal text-white shadow-sm'
                    : 'text-charcoal-muted hover:text-[#B54E35] hover:bg-sand/80'
                }`}
              >
                {item.label}
              </button>
            );
          })}
        </nav>
      </div>
    </header>
  );
};

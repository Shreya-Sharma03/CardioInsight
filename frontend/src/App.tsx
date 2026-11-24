import React, { useState, useEffect } from 'react';
import { ActivePage, HealthResponse } from './types';
import { fetchHealth } from './api/client';
import { Navbar } from './components/Navbar';
import { HomePage } from './pages/HomePage';
import { PredictionStudioPage } from './pages/PredictionStudioPage';
import { ModelInsightsPage } from './pages/ModelInsightsPage';

export const App: React.FC = () => {
  const [activePage, setActivePage] = useState<ActivePage>('home');
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthLoading, setHealthLoading] = useState<boolean>(true);

  useEffect(() => {
    let isMounted = true;
    async function checkHealth() {
      try {
        const h = await fetchHealth();
        if (isMounted) setHealth(h);
      } catch (err) {
        console.warn('Backend API connection check failed:', err);
        if (isMounted) setHealth(null);
      } finally {
        if (isMounted) setHealthLoading(false);
      }
    }
    checkHealth();
    const interval = setInterval(checkHealth, 30000);
    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, []);

  return (
    <div className="min-h-screen bg-canvas text-charcoal flex flex-col font-sans selection:bg-sage-light selection:text-sage-dark">
      {/* Top App Navbar */}
      <Navbar
        activePage={activePage}
        onNavigate={setActivePage}
        health={health}
        healthLoading={healthLoading}
      />

      {/* Main Content Area */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 pb-12">
        {activePage === 'home' && <HomePage onNavigate={setActivePage} />}
        {activePage === 'prediction' && <PredictionStudioPage />}
        {activePage === 'insights' && <ModelInsightsPage />}
      </main>

      {/* Product Footer */}
      <footer className="border-t border-sandBorder bg-sand/40 py-6">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between gap-4 text-xs text-charcoal-muted">
          <div>
            <span className="font-semibold text-charcoal">CardioInsight</span> · ECG-Based Heart Failure Risk Prediction using Adaptive Reliability Fusion
          </div>
          <div className="flex items-center gap-4">
            <span className="font-mono text-[11px] text-charcoal-subtle">
              Checkpoint: Epoch 7 Best · 2,514,819 Params
            </span>
            <span className="text-sandBorder">|</span>
            <span>Research Decision-Support Prototype</span>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default App;

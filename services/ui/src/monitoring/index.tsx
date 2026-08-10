import { createRoot } from 'react-dom/client';
import { CostAnalysis } from './cost_analysis';

function HelloReact() {
  return <p style={{ fontFamily: 'sans-serif', padding: '1rem' }}>Hello React (monitoring)</p>;
}

const helloEl = document.getElementById('monitoring-react-root');
if (helloEl) createRoot(helloEl).render(<HelloReact />);

const costEl = document.getElementById('cost-analysis-root');
if (costEl) {
  const monitoringBaseUrl = costEl.dataset.monitoringBaseUrl ?? '';
  const batchBaseUrl = costEl.dataset.batchBaseUrl ?? '';
  createRoot(costEl).render(<CostAnalysis monitoringBaseUrl={monitoringBaseUrl} batchBaseUrl={batchBaseUrl} />);
}

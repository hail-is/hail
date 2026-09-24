import { createRoot } from 'react-dom/client';
import { CostAnalysis } from './cost_analysis';

const costEl = document.getElementById('cost-analysis-root');
if (costEl) {
  const monitoringBaseUrl = costEl.dataset.monitoringBaseUrl ?? '';
  const batchBaseUrl = costEl.dataset.batchBaseUrl ?? '';
  createRoot(costEl).render(<CostAnalysis monitoringBaseUrl={monitoringBaseUrl} batchBaseUrl={batchBaseUrl} />);
}

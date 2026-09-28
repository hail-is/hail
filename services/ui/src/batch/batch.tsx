import { createRoot } from 'react-dom/client';
import { BatchDetailsPage } from './components/BatchDetailsPage';

const rootEl = document.getElementById('batch-details-root');
if (rootEl) {
  const basePath = rootEl.dataset.basePath ?? '';
  const batchId = rootEl.dataset.batchId ?? '';

  const root = createRoot(rootEl);
  root.render(
    <BatchDetailsPage
      basePath={basePath}
      batchId={batchId}
    />
  );
}

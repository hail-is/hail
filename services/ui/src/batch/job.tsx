import { createRoot } from 'react-dom/client';
import { JobPage } from './components/JobPage';

const rootEl = document.getElementById('job-details-root');
if (rootEl) {
  const basePath = rootEl.dataset.basePath ?? '';
  const batchId = rootEl.dataset.batchId ?? '';
  const jobId = rootEl.dataset.jobId ?? '';

  const root = createRoot(rootEl);
  root.render(
    <JobPage
      basePath={basePath}
      batchId={batchId}
      jobId={jobId}
    />
  );
}

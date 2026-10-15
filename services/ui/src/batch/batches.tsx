import { createRoot } from 'react-dom/client';
import { BatchesPage } from './components/BatchesPage';

const rootEl = document.getElementById('batches-root');
if (rootEl) {
  const basePath = rootEl.dataset.basePath ?? '';
  const username = rootEl.dataset.username ?? '';

  const root = createRoot(rootEl);
  root.render(
    <BatchesPage
      basePath={basePath}
      username={username}
    />
  );
}

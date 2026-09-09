import { createRoot } from 'react-dom/client';
import { DriverDashboard } from './DriverDashboard';

const el = document.getElementById('batch-driver-react-root');
if (el) {
  createRoot(el).render(
    <DriverDashboard
      basePath={el.dataset.basePath ?? ''}
    />,
  );
}

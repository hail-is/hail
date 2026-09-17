import { createRoot } from 'react-dom/client';
import { UserPage } from './components/UserPage';

const rootEl = document.getElementById('user-details-root');
if (rootEl) {
  const authBaseUrl = rootEl.dataset.authBaseUrl ?? '';
  const username = rootEl.dataset.username ?? '';
  const gsaDisplayName = rootEl.dataset.gsaDisplayName ?? '';
  const trialBpName = rootEl.dataset.trialBpName ?? '';

  const root = createRoot(rootEl);
  root.render(
    <UserPage
      authBaseUrl={authBaseUrl}
      username={username}
      gsaDisplayName={gsaDisplayName}
      trialBpName={trialBpName}
    />
  );
}

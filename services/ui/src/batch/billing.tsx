import { createRoot } from 'react-dom/client';
import { BillingProjectsPage } from './billing/BillingProjectsPage';
import { BillingProjectPage } from './billing/BillingProjectPage';
import { BillingPage } from './billing/BillingPage';
import { QuotesPage } from './billing/QuotesPage';
import { QuotePage } from './billing/QuotePage';

const handlers: Record<string, () => void> = {
  'billing-root': () => {
    const el = document.getElementById('billing-root')!;
    const basePath = el.dataset.basePath ?? '';
    const isGlobalBm = el.dataset.isGlobalBm === 'true';
    const username = el.dataset.username ?? '';
    const params = new URLSearchParams(window.location.search);
    const initialStart = params.get('start') ?? '';
    const initialEnd = params.get('end') ?? '';
    createRoot(el).render(
      <BillingPage basePath={basePath} isGlobalBm={isGlobalBm} username={username} initialStart={initialStart} initialEnd={initialEnd} />
    );
  },
  'billing-projects-root': () => {
    const el = document.getElementById('billing-projects-root')!;
    const basePath = el.dataset.basePath ?? '';
    const isGlobalBm = el.dataset.isGlobalBm === 'true';
    const canCreateBp = el.dataset.canCreateBp === 'true';
    const canCreateQuotes = el.dataset.canCreateQuotes === 'true';
    createRoot(el).render(
      <BillingProjectsPage
        basePath={basePath}
        isGlobalBm={isGlobalBm}
        canCreateBp={canCreateBp}
        canCreateQuotes={canCreateQuotes}
      />
    );
  },
  'billing-project-root': () => {
    const el = document.getElementById('billing-project-root')!;
    const basePath = el.dataset.basePath ?? '';
    const bpName = el.dataset.bpName ?? '';
    createRoot(el).render(
      <BillingProjectPage basePath={basePath} bpName={bpName} />
    );
  },
  'quotes-root': () => {
    const el = document.getElementById('quotes-root')!;
    const basePath = el.dataset.basePath ?? '';
    const canCreate = el.dataset.canCreate === 'true';
    createRoot(el).render(<QuotesPage basePath={basePath} canCreate={canCreate} />);
  },
  'quote-root': () => {
    const el = document.getElementById('quote-root')!;
    const basePath = el.dataset.basePath ?? '';
    const quoteName = el.dataset.quoteName ?? '';
    createRoot(el).render(
      <QuotePage basePath={basePath} quoteName={quoteName} />
    );
  },
};

for (const [id, mount] of Object.entries(handlers)) {
  if (document.getElementById(id)) {
    mount();
    break;
  }
}

import { useEffect, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { createCiApi, type DeveloperBranchStatus, type PrConfig, type WatchedBranchSummary } from '../shared/ciApi';
import { PrTable } from './components/PrTable';
import { matchesFilter, nextSort, PR_FILTERS, sortPrs, type PrFilter, type SortState } from './components/prListView';
import { matchesSearch, statusColumnNames } from './components/prTableModel';

function errorMessage(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}

function PrsPage({ basePath, wbIndex }: { basePath: string; wbIndex: number }): JSX.Element {
  const [wb, setWb] = useState<WatchedBranchSummary | null>(null);
  const [prs, setPrs] = useState<PrConfig[] | null>(null);
  const [branchStatus, setBranchStatus] = useState<DeveloperBranchStatus | undefined>(undefined);
  const [error, setError] = useState<string | null>(null);
  const [search, setSearch] = useState('');
  const [filter, setFilter] = useState<PrFilter>('all');
  const [sort, setSort] = useState<SortState | null>(null);
  // GitHub username of the signed-in user, for the Mine / Waiting on me filters. Null until
  // loaded, and stays null for anyone who isn't a CI developer, which hides those filters.
  const [viewer, setViewer] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    createCiApi(basePath).getDeveloper('me')
      .then((dev) => { if (!cancelled) setViewer(dev.gh_username); })
      .catch((e: unknown) => { console.warn('Could not look up the signed-in developer', e); });
    return () => { cancelled = true; };
  }, [basePath]);

  // The shell only knows the branch index from the URL; the name the PRs endpoint needs comes
  // from /watched_branches, as on the PR details page.
  useEffect(() => {
    let cancelled = false;
    const api = createCiApi(basePath);
    api.getWatchedBranches()
      .then((r) => {
        const entry = Number.isInteger(wbIndex) && wbIndex >= 0 ? r.branches.at(wbIndex) : undefined;
        if (entry === undefined) throw new Error(`No watched branch at index ${wbIndex}`);
        if (cancelled) return null;
        setWb(entry);
        // Only the Merge column's queue details need this, so the table renders without it
        api.getDeveloperStatus()
          .then((status) => {
            if (!cancelled) setBranchStatus(status.branches.find((b) => b.branch === entry.branch));
          })
          .catch((e: unknown) => { console.warn('Could not load CI developer status', e); });
        return api.getWatchedBranchPrs(entry.branch);
      })
      .then((branchPrs) => {
        if (!cancelled && branchPrs !== null) setPrs(branchPrs);
      })
      .catch((e: unknown) => {
        if (!cancelled) setError(errorMessage(e));
      });
    return () => { cancelled = true; };
  }, [basePath, wbIndex]);

  let body: JSX.Element;
  if (error) {
    body = <p className="text-red-600">Error loading PRs: {error}</p>;
  } else if (wb === null || prs === null) {
    body = <p className="text-sm text-zinc-500">Loading&hellip;</p>;
  } else if (prs.length === 0) {
    body = <p className="text-sm text-zinc-500">No PRs.</p>;
  } else {
    const statusNames = statusColumnNames(prs);
    const nowMs = Date.now();
    const shown = sortPrs(
      prs.filter((pr) => matchesFilter(pr, filter, viewer) && matchesSearch(pr, statusNames, search, branchStatus, nowMs)),
      sort,
      { branch: branchStatus, nowMs },
    );
    body = shown.length === 0
      ? <p className="text-sm text-zinc-500">No PRs match these filters.</p>
      : (
        <PrTable
          prs={shown}
          statusNames={statusNames}
          repo={wb.repo}
          basePath={basePath}
          wbIndex={wbIndex}
          branchStatus={branchStatus}
          sort={sort}
          onSort={(column) => { setSort((current) => nextSort(current, column)); }}
        />
      );
  }

  return (
    <div className="pb-8">
      <nav className="flex items-center gap-2 text-xl font-light text-zinc-500 flex-wrap">
        <a href={`${basePath}/`} className="hover:text-sky-600">CI</a>
        <span className="text-zinc-300">›</span>
        <span className="text-zinc-800">{wb?.branch_fqn ?? '…'} PRs{prs ? ` (${prs.length})` : ''}</span>
      </nav>

      <div className="mt-4 mb-4 flex flex-wrap items-center gap-3">
        <label htmlFor="prs-search" className="sr-only">Search PRs</label>
        <input
          id="prs-search"
          type="search"
          value={search}
          onChange={(e) => { setSearch(e.target.value); }}
          placeholder="Search terms…"
          className="w-full max-w-md px-3 py-1.5 text-sm border border-zinc-300 rounded focus:outline-none focus:ring-2 focus:ring-sky-500"
        />
        <div role="group" aria-label="Filter PRs" className="flex flex-wrap gap-1.5">
          {PR_FILTERS.filter((f) => !f.needsViewer || viewer !== null).map((f) => {
            const active = f.value === filter;
            return (
              <button
                type="button"
                key={f.value}
                aria-pressed={active}
                onClick={() => { setFilter(f.value); }}
                className={`text-xs px-2 py-0.5 rounded-full border transition-colors ${active ? 'bg-sky-500 border-sky-500 text-white' : 'border-zinc-300 text-zinc-500 hover:border-sky-400 hover:text-sky-600 bg-white'}`}
              >
                {f.label}
              </button>
            );
          })}
        </div>
      </div>

      {body}
    </div>
  );
}

const container = document.getElementById('prs-root');
if (container) {
  const basePath = container.dataset.basePath ?? '';
  const wbIndex = parseInt(container.dataset.watchedBranchIndex ?? '', 10);
  createRoot(container).render(<PrsPage basePath={basePath} wbIndex={wbIndex} />);
}

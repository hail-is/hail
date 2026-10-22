import { useEffect, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { createCiApi, type PrConfig, type WatchedBranchSummary } from '../shared/ciApi';
import { PrTable } from './components/PrTable';
import { matchesSearch, statusColumnNames } from './components/prTableModel';

function errorMessage(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}

function PrsPage({ basePath, wbIndex }: { basePath: string; wbIndex: number }): JSX.Element {
  const [wb, setWb] = useState<WatchedBranchSummary | null>(null);
  const [prs, setPrs] = useState<PrConfig[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [search, setSearch] = useState('');
  const searchRef = useRef<HTMLInputElement>(null);

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

  // `/` jumps to the search box, as on the classic page.
  useEffect(() => {
    const onKeyUp = (e: KeyboardEvent): void => {
      const t = e.target;
      if (e.key === '/' && !(t instanceof HTMLElement && (t.matches('input, textarea') || t.isContentEditable))) {
        searchRef.current?.focus();
      }
    };
    document.addEventListener('keyup', onKeyUp);
    return () => { document.removeEventListener('keyup', onKeyUp); };
  }, []);

  let body: JSX.Element;
  if (error) {
    body = <p className="text-red-600">Error loading PRs: {error}</p>;
  } else if (wb === null || prs === null) {
    body = <p className="text-sm text-zinc-500">Loading&hellip;</p>;
  } else if (prs.length === 0) {
    body = <p className="text-sm text-zinc-500">No PRs.</p>;
  } else {
    const statusNames = statusColumnNames(prs);
    const shown = prs.filter((pr) => matchesSearch(pr, statusNames, search));
    body = shown.length === 0
      ? <p className="text-sm text-zinc-500">No PRs match &ldquo;{search}&rdquo;.</p>
      : <PrTable prs={shown} statusNames={statusNames} repo={wb.repo} basePath={basePath} wbIndex={wbIndex} />;
  }

  return (
    <div className="pb-8">
      <nav className="flex items-center gap-2 text-xl font-light text-zinc-500 flex-wrap">
        <a href={`${basePath}/`} className="hover:text-sky-600">CI</a>
        <span className="text-zinc-300">›</span>
        <span className="text-zinc-800">{wb?.branch_fqn ?? '…'} PRs{prs ? ` (${prs.length})` : ''}</span>
      </nav>

      <div className="mt-4 mb-4">
        <label htmlFor="prs-search" className="sr-only">Search PRs</label>
        <input
          id="prs-search"
          ref={searchRef}
          type="search"
          value={search}
          onChange={(e) => { setSearch(e.target.value); }}
          placeholder="Search terms… (press / to focus)"
          className="w-full max-w-md px-3 py-1.5 text-sm border border-zinc-300 rounded focus:outline-none focus:ring-2 focus:ring-sky-500"
        />
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

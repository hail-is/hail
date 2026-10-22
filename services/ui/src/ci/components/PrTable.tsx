import type { PrConfig } from '../../shared/ciApi';
import { BuildStateIcon } from './BuildStateIcon';
import { ReviewStateIcon } from './ReviewStateIcon';
import { buildStateLabel, ghStatuses, isBuildOutOfDate } from './prTableModel';

// GitHub's mark (octicons `mark-github`, 16px).
function GitHubIcon(): JSX.Element {
  return (
    <svg aria-hidden="true" viewBox="0 0 16 16" className="w-4 h-4" fill="currentColor">
      <path d="M8 0c4.42 0 8 3.58 8 8a8.013 8.013 0 0 1-5.45 7.59c-.4.08-.55-.17-.55-.38 0-.27.01-1.13.01-2.2 0-.75-.25-1.23-.54-1.48 1.78-.2 3.65-.88 3.65-3.95 0-.88-.31-1.59-.82-2.15.08-.2.36-1.02-.08-2.12 0 0-.67-.22-2.2.82-.64-.18-1.32-.27-2-.27-.68 0-1.36.09-2 .27-1.53-1.03-2.2-.82-2.2-.82-.44 1.1-.16 1.92-.08 2.12-.51.56-.82 1.28-.82 2.15 0 3.06 1.86 3.75 3.64 3.95-.23.2-.44.55-.51 1.07-.46.21-1.61.55-2.33-.66-.15-.24-.6-.83-1.23-.82-.67.01-.27.38.01.53.34.19.73.9.82 1.13.16.45.68 1.31 2.69.94 0 .67.01 1.3.01 1.49 0 .21-.15.45-.55.38A7.995 7.995 0 0 1 0 8c0-4.42 3.58-8 8-8Z" />
    </svg>
  );
}

// Table of a watched branch's open PRs. `statusNames` is the column set for the whole branch,
// so filtering rows doesn't add or drop columns.
export function PrTable({ prs, statusNames, repo, basePath, wbIndex }: {
  prs: PrConfig[];
  statusNames: string[];
  repo: string;
  basePath: string;
  wbIndex: number;
}): JSX.Element {
  const th = 'px-3 py-1 font-medium';
  const td = 'px-3 py-1 align-top';
  return (
    <>
      <div className="overflow-x-auto">
        <table className="w-full text-sm border border-zinc-200">
          <thead className="bg-zinc-100 text-left text-xs uppercase text-zinc-500">
            <tr>
              <th rowSpan={2} className={th}>PR</th>
              <th rowSpan={2} className={th}>Build state</th>
              {statusNames.length > 0 && (
                <th colSpan={statusNames.length} className={`${th} text-center border-x border-zinc-200`}>GitHub statuses</th>
              )}
              <th rowSpan={2} className={th}>Labels</th>
              <th rowSpan={2} className={th}>Branch</th>
              <th rowSpan={2} className={th}>Review state</th>
              <th rowSpan={2} className={th}>Author</th>
            </tr>
            <tr>
              {statusNames.map((name) => (
                <th key={name} className={`${th} normal-case`}>{name}</th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-zinc-100">
            {prs.map((pr) => {
              const statuses = ghStatuses(pr);
              return (
                <tr key={pr.number} className="hover:bg-zinc-50">
                  <td className={td}>
                    <a
                      href={`https://github.com/${repo}/pull/${pr.number}`}
                      target="_blank"
                      rel="noopener noreferrer"
                      title="Open on GitHub"
                      className="text-sky-600 hover:underline"
                    >
                      <span className="inline-block align-text-bottom mr-1.5 text-zinc-500"><GitHubIcon /></span>
                      {pr.title} <span className="text-zinc-400">#{pr.number}</span>
                    </a>
                  </td>
                  <td className={`${td} whitespace-nowrap`}>
                    <a
                      href={`${basePath}/watched_branches/${wbIndex}/pr/${pr.number}`}
                      title="CI details"
                      className="group inline-flex items-center gap-1.5 text-sky-600"
                    >
                      <img src={`${basePath}/common_static/hail_logo_sq.ico`} alt="" width={16} height={16} className="w-4 h-4" />
                      <BuildStateIcon state={pr.build_state} />
                      <span className="group-hover:underline">{buildStateLabel(pr)}</span>
                    </a>
                  </td>
                  {statusNames.map((name) => {
                    const status = statuses.get(name) ?? '';
                    return (
                      <td key={name} className={`${td} whitespace-nowrap`}>
                        {status !== '' && (
                          <span className="inline-flex items-center gap-1.5" title={`${name}: ${status}`}>
                            <BuildStateIcon state={status} />
                            {status}
                          </span>
                        )}
                      </td>
                    );
                  })}
                  <td className={td}>{pr.labels.join(', ')}</td>
                  <td className={`${td} font-mono text-xs`}>{pr.source_branch_name}</td>
                  <td className={`${td} whitespace-nowrap`}>
                    {pr.review_state !== null && (
                      <span className="inline-flex items-center gap-1.5">
                        <ReviewStateIcon state={pr.review_state} />
                        {pr.review_state}
                      </span>
                    )}
                  </td>
                  <td className={td}>
                  <a
                    href={`https://github.com/${repo}/pulls/${encodeURIComponent(pr.author)}`}
                    target="_blank"
                    rel="noopener noreferrer"
                    title={`${pr.author}'s PRs on GitHub`}
                    className="text-sky-600 hover:underline"
                  >
                    <span className="inline-block align-text-bottom mr-1.5 text-zinc-500"><GitHubIcon /></span>
                    {pr.author}
                  </a>
                </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      {prs.some(isBuildOutOfDate) && (
        <p className="mt-2 text-xs text-zinc-500">
          * Built or building against an older commit of the target branch. CI rebuilds it against the latest
          commit when it becomes the merge candidate.
        </p>
      )}
    </>
  );
}

import type { DeveloperBranchStatus, PrConfig } from '../../shared/ciApi';
import type { ReactNode } from 'react';
import { BuildStateIcon } from './BuildStateIcon';
import { LabelChip } from './LabelChip';
import { ReviewStateIcon } from './ReviewStateIcon';
import {
  buildStateLabel,
  formatAge,
  ghStatuses,
  type JourneyMark,
  journeyMarks,
  labelsCell,
  mergeCell,
  reviewSlots,
  type StageCell,
  upToDateCell,
} from './prTableModel';
import { visibleLabels } from './labelModel';
import type { SortColumn, SortState } from './prListView';

// GitHub's mark (octicons `mark-github`, 16px).
function GitHubIcon(): JSX.Element {
  return (
    <svg aria-hidden="true" viewBox="0 0 16 16" className="w-4 h-4" fill="currentColor">
      <path d="M8 0c4.42 0 8 3.58 8 8a8.013 8.013 0 0 1-5.45 7.59c-.4.08-.55-.17-.55-.38 0-.27.01-1.13.01-2.2 0-.75-.25-1.23-.54-1.48 1.78-.2 3.65-.88 3.65-3.95 0-.88-.31-1.59-.82-2.15.08-.2.36-1.02-.08-2.12 0 0-.67-.22-2.2.82-.64-.18-1.32-.27-2-.27-.68 0-1.36.09-2 .27-1.53-1.03-2.2-.82-2.2-.82-.44 1.1-.16 1.92-.08 2.12-.51.56-.82 1.28-.82 2.15 0 3.06 1.86 3.75 3.64 3.95-.23.2-.44.55-.51 1.07-.46.21-1.61.55-2.33-.66-.15-.24-.6-.83-1.23-.82-.67.01-.27.38.01.53.34.19.73.9.82 1.13.16.45.68 1.31 2.69.94 0 .67.01 1.3.01 1.49 0 .21-.15.45-.55.38A7.995 7.995 0 0 1 0 8c0-4.42 3.58-8 8-8Z" />
    </svg>
  );
}

function StageCellView({ cell, nowMs }: { cell: StageCell; nowMs: number }): JSX.Element {
  if (cell.status === 'na') {
    return cell.text === ''
      ? <span className="text-zinc-300" title={cell.detail}>&mdash;</span>
      : <span className="text-zinc-400" title={cell.detail}>{cell.text}</span>;
  }
  return (
    <span className="inline-flex items-center gap-1.5" title={cell.detail}>
      <ReviewStateIcon status={cell.status} />
      <span>
        {cell.text}
        {cell.requestedAt !== undefined && (
          <span className="text-zinc-400" title={`requested ${cell.requestedAt}`}> &middot; {formatAge(cell.requestedAt, nowMs)}</span>
        )}
      </span>
    </span>
  );
}

// A column header that sorts the table when clicked; the arrow shows the current direction.
function SortHeader({ column, sort, onSort, className, rowSpan, children }: {
  column: SortColumn;
  sort: SortState | null;
  onSort: (_column: SortColumn) => void;
  className: string;
  rowSpan?: number;
  children: ReactNode;
}): JSX.Element {
  const dir = sort?.column === column ? sort.dir : null;
  let ariaSort: 'ascending' | 'descending' | 'none' = 'none';
  if (dir === 'asc') ariaSort = 'ascending';
  else if (dir === 'desc') ariaSort = 'descending';
  return (
    <th rowSpan={rowSpan} className={className} aria-sort={ariaSort}>
      <button
        type="button"
        onClick={() => { onSort(column); }}
        title="Sort (click again to reverse, a third time to reset)"
        className="inline-flex items-center gap-0.5 uppercase hover:text-zinc-800"
      >
        {children}
        <span aria-hidden="true" className={`material-symbols-outlined text-sm ${dir === null ? 'invisible' : ''}`}>
          {dir === 'desc' ? 'arrow_downward' : 'arrow_upward'}
        </span>
      </button>
    </th>
  );
}

// Each merge journey step starts with a left border, so the steps read left to right.
const STEP = 'border-l border-zinc-200';

// Background for a journey cell: green while every cell up to it is done; the first cell that
// isn't is tinted red (failed) or amber (waiting).
const MARK_CLASS: Record<JourneyMark, string> = {
  done: 'bg-green-50',
  fail: 'bg-red-50',
  pending: 'bg-amber-50',
  todo: '',
};

// The `›` on the border into a journey cell, shown only when the cell before it (`after`) is done.
function StepArrow({ after }: { after: JourneyMark }): JSX.Element | null {
  if (after !== 'done') return null;
  return (
    <span
      aria-hidden="true"
      className="material-symbols-outlined absolute left-0 top-1/2 -translate-x-1/2 -translate-y-1/2 text-2xl leading-none text-green-600"
    >
      chevron_right
    </span>
  );
}

// Table of a watched branch's open PRs. The merge journey columns run in the order CI needs them
// (build, reviews, labels, up to date, merge), so the first step that isn't green is what's
// holding a PR up. `statusNames` is the column set for the whole branch, so filtering rows doesn't
// add or drop columns. `branchStatus` adds queue position and merge progress to the Merge step;
// it's optional so the table still renders without it. Rows are shown in the order given; `sort`
// only marks which header is active.
export function PrTable({ prs, statusNames, repo, basePath, wbIndex, branchStatus, sort, onSort }: {
  prs: PrConfig[];
  statusNames: string[];
  repo: string;
  basePath: string;
  wbIndex: number;
  branchStatus?: DeveloperBranchStatus;
  sort: SortState | null;
  onSort: (_column: SortColumn) => void;
}): JSX.Element {
  const th = 'px-3 py-1 font-medium';
  const sortable = { sort, onSort };
  const td = 'px-3 py-1 align-top';
  const nowMs = Date.now();
  return (
    <>
      <div className="overflow-x-auto">
        <table className="w-full text-sm border border-zinc-200">
          <thead className="bg-zinc-100 text-left text-xs uppercase text-zinc-500">
            <tr>
              <SortHeader column="pr" rowSpan={3} className={th} {...sortable}>PR</SortHeader>
              <th colSpan={6 + statusNames.length} className={`${th} ${STEP} text-center`}>Merge journey</th>
              <SortHeader column="branch" rowSpan={3} className={`${th} ${STEP}`} {...sortable}>Branch</SortHeader>
              <SortHeader column="author" rowSpan={3} className={th} {...sortable}>Author</SortHeader>
            </tr>
            <tr>
              <th colSpan={1 + statusNames.length} className={`${th} ${STEP} text-center`}>Build</th>
              <SortHeader column="review1" rowSpan={2} className={`${th} ${STEP}`} {...sortable}>Review 1</SortHeader>
              <SortHeader column="review2" rowSpan={2} className={`${th} ${STEP}`} {...sortable}>Review 2</SortHeader>
              <SortHeader column="labels" rowSpan={2} className={`${th} ${STEP}`} {...sortable}>Labels</SortHeader>
              <SortHeader column="uptodate" rowSpan={2} className={`${th} ${STEP}`} {...sortable}>Up to date</SortHeader>
              <SortHeader column="merge" rowSpan={2} className={`${th} ${STEP}`} {...sortable}>Merge</SortHeader>
            </tr>
            <tr>
              <SortHeader column="build" className={`${th} ${STEP}`} {...sortable}>CI</SortHeader>
              {statusNames.map((name) => (
                <SortHeader key={name} column={`status:${name}`} className={`${th} [&_button]:normal-case`} {...sortable}>{name}</SortHeader>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-zinc-100">
            {prs.map((pr) => {
              const statuses = ghStatuses(pr);
              const [review1, review2] = reviewSlots(pr);
              // Journey cells: 0 = CI, then the statuses, then Review 1 at `r`, ..., Merge at `r + 4`.
              const marks = journeyMarks(pr, statusNames, branchStatus, nowMs);
              const r = 1 + statusNames.length;
              const step = (i: number): string => `${td} ${STEP} ${MARK_CLASS[marks[i]]} relative`;
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
                  <td className={`${td} ${STEP} ${MARK_CLASS[marks[0]]} whitespace-nowrap`}>
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
                  {statusNames.map((name, i) => {
                    const status = statuses.get(name) ?? '';
                    return (
                      <td key={name} className={`${td} ${MARK_CLASS[marks[1 + i]]} relative whitespace-nowrap`}>
                        <StepArrow after={marks[i]} />
                        {status !== '' && (
                          <span className="inline-flex items-center gap-1.5" title={`${name}: ${status}`}>
                            <BuildStateIcon state={status} />
                            {status}
                          </span>
                        )}
                      </td>
                    );
                  })}
                  <td className={step(r)}><StepArrow after={marks[r - 1]} /><StageCellView cell={review1} nowMs={nowMs} /></td>
                  <td className={step(r + 1)}><StepArrow after={marks[r]} /><StageCellView cell={review2} nowMs={nowMs} /></td>
                  <td className={step(r + 2)} title={labelsCell(pr).detail}><StepArrow after={marks[r + 1]} />
                    <span className="inline-flex flex-wrap gap-1">
                      {visibleLabels(pr.labels).map((label) => <LabelChip key={label} label={label} />)}
                    </span>
                  </td>
                  <td className={`${step(r + 3)} whitespace-nowrap`}><StepArrow after={marks[r + 2]} /><StageCellView cell={upToDateCell(pr)} nowMs={nowMs} /></td>
                  <td className={step(r + 4)}><StepArrow after={marks[r + 3]} /><StageCellView cell={mergeCell(pr, branchStatus, nowMs)} nowMs={nowMs} /></td>
                  <td className={`${td} ${STEP} font-mono text-xs`}>{pr.source_branch_name}</td>
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
    </>
  );
}

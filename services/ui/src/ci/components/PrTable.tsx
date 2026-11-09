import type { ReactNode } from 'react';
import type { DeveloperBranchStatus, PrConfig, PrStack, PrStackEntry } from '../../shared/ciApi';
import { BuildStateIcon } from './BuildStateIcon';
import { LabelChip } from './LabelChip';
import { ReviewStateIcon } from './ReviewStateIcon';
import {
  buildStateLabel,
  buildTally,
  formatAge,
  ghStatuses,
  type JourneyMark,
  journeyMarks,
  labelsCell,
  mergeCell,
  reviewSlots,
  stageTally,
  type StageCell,
  type StageStatus,
  statusTally,
  type TallyItem,
  upToDateCell,
} from './prTableModel';
import { visibleLabels } from './labelModel';
import type { ListItem, SortColumn, SortState, StackRow } from './prListView';

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

// Stack members' titles sit one step in from their summary row.
const STACK_INDENT = 'pl-8';

// One listed PR. A stack member's title is indented under the stack's summary row.
function PrRow({ pr, statusNames, repo, basePath, wbIndex, branchStatus, nowMs, inStack = false }: {
  pr: PrConfig;
  statusNames: string[];
  repo: string;
  basePath: string;
  wbIndex: number;
  branchStatus?: DeveloperBranchStatus;
  nowMs: number;
  inStack?: boolean;
}): JSX.Element {
  const td = 'px-3 py-1 align-top';
  const statuses = ghStatuses(pr);
  const [review1, review2] = reviewSlots(pr);
  // Journey cells: 0 = CI, then the statuses, then Review 1 at `r`, ..., Merge at `r + 4`.
  const marks = journeyMarks(pr, statusNames, branchStatus, nowMs);
  const r = 1 + statusNames.length;
  const step = (i: number): string => `${td} ${STEP} ${MARK_CLASS[marks[i]]} relative`;
  return (
    <tr className="hover:bg-zinc-50">
      <td className={`${td} ${inStack ? STACK_INDENT : ''}`}>
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
        {pr.display_only ? (
          // Not built, so there are no CI details to link to.
          <span className="inline-flex items-center gap-1.5 text-zinc-500">
            <BuildStateIcon state={pr.build_state} />
            {buildStateLabel(pr)}
          </span>
        ) : (
          <a
            href={`${basePath}/watched_branches/${wbIndex}/pr/${pr.number}`}
            title="CI details"
            className="group inline-flex items-center gap-1.5 text-sky-600"
          >
            <img src={`${basePath}/common_static/hail_logo_sq.ico`} alt="" width={16} height={16} className="w-4 h-4" />
            <BuildStateIcon state={pr.build_state} />
            <span className="group-hover:underline">{buildStateLabel(pr)}</span>
          </a>
        )}
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
      <td className={step(r + 2)} title={labelsCell(pr).detail}>
        <StepArrow after={marks[r + 1]} />
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
}

// A merged, closed or unlisted stack entry: title and state only, muted.
function StackEntryRow({ entry, columns, repo, basePath, wbIndex }: {
  entry: PrStackEntry;
  columns: number;
  repo: string;
  basePath: string;
  wbIndex: number;
}): JSX.Element {
  const td = 'px-3 py-1 align-top';
  return (
    <tr className="text-zinc-400">
      <td className={`${td} ${STACK_INDENT}`}>
        <a
          href={`https://github.com/${repo}/pull/${entry.number}`}
          target="_blank"
          rel="noopener noreferrer"
          title="Open on GitHub"
          className="hover:underline"
        >
          <span className="inline-block align-text-bottom mr-1.5"><GitHubIcon /></span>
          {entry.title} <span>#{entry.number}</span>
        </a>
      </td>
      <td className={td} colSpan={columns - 1}>
        <a href={`${basePath}/watched_branches/${wbIndex}/pr/${entry.number}`} title="CI details" className="hover:underline">
          {entry.state.toLowerCase()}
        </a>
      </td>
    </tr>
  );
}

function plural(n: number, word: string, pluralWord = `${word}s`): string {
  return `${n} ${n === 1 ? word : pluralWord}`;
}

// A summary cell: counts per state, e.g. "✓ 5 · … 6".
function TallyView({ items }: { items: TallyItem[] }): JSX.Element {
  return (
    <span className="inline-flex flex-wrap items-center gap-x-2 text-zinc-600">
      {items.map(({ kind, state, count }) => {
        const icon = kind === 'stage' ? <ReviewStateIcon status={state as StageStatus} /> : <BuildStateIcon state={state} />;
        return (
          <span key={state} className="inline-flex items-center gap-0.5 whitespace-nowrap" title={`${count} ${state}`}>
            {icon}
            {count}
          </span>
        );
      })}
    </span>
  );
}

// A stack's summary row: name, doc link, size and a toggle for its member rows, then each
// column's states counted across the stack's listed PRs.
function StackSummaryRow({ stack, rows, matchCount, statusNames, branchStatus, nowMs, expanded, onToggle }: {
  stack: PrStack;
  rows: StackRow[];
  matchCount: number | null;
  statusNames: string[];
  branchStatus?: DeveloperBranchStatus;
  nowMs: number;
  expanded: boolean;
  onToggle: () => void;
}): JSX.Element {
  const merged = stack.entries.filter((e) => e.state === 'MERGED').length;
  const prs = rows.flatMap((row) => (row.kind === 'pr' ? [row.pr] : []));
  const td = 'px-3 py-1.5 align-top';
  const step = (cells: StageCell[]): JSX.Element => (
    <td className={`${td} ${STEP}`}><TallyView items={stageTally(cells)} /></td>
  );
  const authors = [...new Set(prs.map((pr) => pr.author))];
  return (
    <tr>
      <td className={td}>
        <span className="inline-flex flex-wrap items-center gap-x-3 gap-y-1">
          <button
            type="button"
            onClick={onToggle}
            aria-expanded={expanded}
            className="inline-flex items-center gap-1 font-semibold text-zinc-800 hover:text-sky-700"
          >
            <span aria-hidden="true" className="material-symbols-outlined text-base">{expanded ? 'expand_more' : 'chevron_right'}</span>
            <span aria-hidden="true" className="material-symbols-outlined text-base text-amber-500">stacks</span>
            {stack.name ?? 'Unnamed stack'}
          </button>
          <span className="text-xs text-zinc-500">
            {plural(stack.entries.length, 'PR')}{merged > 0 && ` · ${merged} merged`}
          </span>
          {matchCount !== null && <span className="text-xs text-sky-700">{plural(matchCount, 'match', 'matches')}</span>}
          {stack.doc_url !== null && (
            <a href={stack.doc_url} target="_blank" rel="noopener noreferrer" className="text-xs text-sky-600 hover:underline">
              reference doc
            </a>
          )}
        </span>
      </td>
      <td className={`${td} ${STEP}`}><TallyView items={buildTally(prs)} /></td>
      {statusNames.map((name) => <td key={name} className={td}><TallyView items={statusTally(prs, name)} /></td>)}
      {step(prs.map((pr) => reviewSlots(pr)[0]))}
      {step(prs.map((pr) => reviewSlots(pr)[1]))}
      {step(prs.map(labelsCell))}
      {step(prs.map(upToDateCell))}
      {step(prs.map((pr) => mergeCell(pr, branchStatus, nowMs)))}
      <td className={`${td} ${STEP}`} />
      <td className={`${td} text-zinc-600`}>{authors.join(', ')}</td>
    </tr>
  );
}

// Table of a watched branch's open PRs, with stacks as collapsible groups. The merge journey
// columns run in the order CI needs them (build, reviews, labels, up to date, merge), so the first
// step that isn't green is what's holding a PR up. `statusNames` is the column set for the whole
// branch, so filtering rows doesn't add or drop columns. `branchStatus` adds queue position and
// merge progress to the Merge step; it's optional so the table still renders without it. Items
// are shown in the order given; `sort` only marks which header is active.
export function PrTable({ items, statusNames, repo, basePath, wbIndex, branchStatus, sort, onSort, expandedStacks, onToggleStack }: {
  items: ListItem[];
  statusNames: string[];
  repo: string;
  basePath: string;
  wbIndex: number;
  branchStatus?: DeveloperBranchStatus;
  sort: SortState | null;
  onSort: (_column: SortColumn) => void;
  expandedStacks: ReadonlySet<number>; // by `stack.number`
  onToggleStack: (_stackNumber: number) => void;
}): JSX.Element {
  const th = 'px-3 py-1 font-medium';
  const sortable = { sort, onSort };
  const nowMs = Date.now();
  // PR, CI, statuses, review 1 / 2, labels, up to date, merge, branch, author
  const columns = 9 + statusNames.length;
  return (
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
        {/* One body per item, so a stack's body can carry the blue border and background round its rows. */}
        {items.map((item) => {
          const rowProps = { statusNames, repo, basePath, wbIndex, branchStatus, nowMs };
          if (item.kind === 'pr') {
            return (
              <tbody key={`pr-${item.pr.number}`} className="border-t border-zinc-100">
                <PrRow pr={item.pr} {...rowProps} />
              </tbody>
            );
          }
          const expanded = expandedStacks.has(item.stack.number);
          return (
            <tbody key={`stack-${item.stack.number}`} className="border-[3px] border-blue-500 bg-blue-50/60 divide-y divide-zinc-100">
              <StackSummaryRow
                stack={item.stack}
                rows={item.rows}
                matchCount={item.matchCount}
                statusNames={statusNames}
                branchStatus={branchStatus}
                nowMs={nowMs}
                expanded={expanded}
                onToggle={() => { onToggleStack(item.stack.number); }}
              />
              {expanded && item.rows.map((row) => (row.kind === 'pr'
                ? <PrRow key={row.pr.number} pr={row.pr} inStack {...rowProps} />
                : <StackEntryRow key={row.entry.number} entry={row.entry} columns={columns} repo={repo} basePath={basePath} wbIndex={wbIndex} />
              ))}
            </tbody>
          );
        })}
      </table>
    </div>
  );
}

import { Fragment, type ReactNode } from 'react';
import type { DeveloperBranchStatus, PrConfig, PrStack, PrStackEntry } from '../../shared/ciApi';
import { BuildStateIcon } from './BuildStateIcon';
import { LabelChip } from './LabelChip';
import { ReviewStateIcon } from './ReviewStateIcon';
import {
  buildStateLabel,
  formatAge,
  ghStatuses,
  isBuildOutOfDate,
  mergeCell,
  reviewSlots,
  type StageCell,
} from './prTableModel';
import { visibleLabels } from './labelModel';
import type { ListItem, SortColumn, SortState } from './prListView';

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
    return cell.text === '' ? <span className="text-zinc-300">&mdash;</span> : <span className="text-zinc-400">{cell.text}</span>;
  }
  return (
    <span className="inline-flex items-center gap-1.5">
      <ReviewStateIcon status={cell.status} />
      <span>
        {cell.text}
        {cell.requestedAt !== undefined && (
          <span className="text-zinc-400" title={`requested ${cell.requestedAt}`}> &middot; {formatAge(cell.requestedAt, nowMs)}</span>
        )}
      </span>
      {cell.labels?.map((label) => <LabelChip key={label} label={label} />)}
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

// One listed PR. `position` is set for a stack member, which is indented under its summary row.
function PrRow({ pr, statusNames, repo, basePath, wbIndex, branchStatus, nowMs, position }: {
  pr: PrConfig;
  statusNames: string[];
  repo: string;
  basePath: string;
  wbIndex: number;
  branchStatus?: DeveloperBranchStatus;
  nowMs: number;
  position?: number;
}): JSX.Element {
  const td = 'px-3 py-1 align-top';
  const inStack = position !== undefined;
  const statuses = ghStatuses(pr);
  const [review1, review2] = reviewSlots(pr);
  return (
    <tr className={`hover:bg-zinc-50 ${inStack ? STACK_MEMBER_ROW : ''}`}>
      <td className={td} style={indentStyle(position)}>
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
      <td className={td}>
        <span className="inline-flex flex-wrap gap-1">
          {visibleLabels(pr.labels).map((label) => <LabelChip key={label} label={label} />)}
        </span>
      </td>
      <td className={`${td} font-mono text-xs`}>{pr.source_branch_name}</td>
      <td className={td}><StageCellView cell={review1} nowMs={nowMs} /></td>
      <td className={td}><StageCellView cell={review2} nowMs={nowMs} /></td>
      <td className={td}><StageCellView cell={mergeCell(pr, branchStatus, nowMs)} nowMs={nowMs} /></td>
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

// A gold rule ties a stack's rows to its summary row.
const STACK_BORDER = 'border-amber-400';
const STACK_MEMBER_ROW = `border-x-2 ${STACK_BORDER} bg-amber-50/30`;

// Stack members are indented by position (1 = nearest the branch).
function indentStyle(position: number | undefined): { paddingLeft: string } | undefined {
  return position === undefined ? undefined : { paddingLeft: `${0.75 + position * 0.75}rem` };
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
    <tr className={`${STACK_MEMBER_ROW} text-zinc-400`}>
      <td className={td} style={indentStyle(entry.position)}>
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

function plural(n: number, word: string): string {
  return `${n} ${word}${n === 1 ? '' : 's'}`;
}

// A stack's summary row: name, doc link, size and a toggle for its member rows.
function StackSummaryRow({ stack, matchCount, columns, expanded, onToggle }: {
  stack: PrStack;
  matchCount: number | null;
  columns: number;
  expanded: boolean;
  onToggle: () => void;
}): JSX.Element {
  const merged = stack.entries.filter((e) => e.state === 'MERGED').length;
  return (
    <tr className={`border-2 ${STACK_BORDER} ${expanded ? 'border-b-0' : ''} bg-amber-50`}>
      <td colSpan={columns} className="px-3 py-1.5">
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
          {matchCount !== null && <span className="text-xs text-sky-700">{plural(matchCount, 'match')}</span>}
          {stack.doc_url !== null && (
            <a href={stack.doc_url} target="_blank" rel="noopener noreferrer" className="text-xs text-sky-600 hover:underline">
              reference doc
            </a>
          )}
        </span>
      </td>
    </tr>
  );
}

// Table of a watched branch's open PRs, with stacks as collapsible groups. `statusNames` is the
// column set for the whole branch, so filtering rows doesn't add or drop columns. `branchStatus`
// adds queue position and merge progress to the Merge column; it's optional so the table still
// renders without it. Items are shown in the order given; `sort` only marks which header is active.
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
  // PR, build state, statuses, labels, branch, review 1 / 2, merge, author
  const columns = 8 + statusNames.length;
  const listedPrs = items.flatMap((item) => (item.kind === 'pr'
    ? [item.pr]
    : item.rows.flatMap((row) => (row.kind === 'pr' ? [row.pr] : []))));
  return (
    <>
      <div className="overflow-x-auto">
        <table className="w-full text-sm border border-zinc-200">
          <thead className="bg-zinc-100 text-left text-xs uppercase text-zinc-500">
            <tr>
              <SortHeader column="pr" rowSpan={2} className={th} {...sortable}>PR</SortHeader>
              <SortHeader column="build" rowSpan={2} className={th} {...sortable}>Build state</SortHeader>
              {statusNames.length > 0 && (
                <th colSpan={statusNames.length} className={`${th} text-center border-x border-zinc-200`}>GitHub statuses</th>
              )}
              <SortHeader column="labels" rowSpan={2} className={th} {...sortable}>Labels</SortHeader>
              <SortHeader column="branch" rowSpan={2} className={th} {...sortable}>Branch</SortHeader>
              <SortHeader column="review1" rowSpan={2} className={th} {...sortable}>Review 1</SortHeader>
              <SortHeader column="review2" rowSpan={2} className={th} {...sortable}>Review 2</SortHeader>
              <SortHeader column="merge" rowSpan={2} className={th} {...sortable}>Merge</SortHeader>
              <SortHeader column="author" rowSpan={2} className={th} {...sortable}>Author</SortHeader>
            </tr>
            <tr>
              {statusNames.map((name) => (
                <SortHeader key={name} column={`status:${name}`} className={`${th} [&_button]:normal-case`} {...sortable}>{name}</SortHeader>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-zinc-100">
            {items.map((item) => {
              const rowProps = { statusNames, repo, basePath, wbIndex, branchStatus, nowMs };
              if (item.kind === 'pr') return <PrRow key={`pr-${item.pr.number}`} pr={item.pr} {...rowProps} />;
              const expanded = expandedStacks.has(item.stack.number);
              return (
                <Fragment key={`stack-${item.stack.number}`}>
                  <StackSummaryRow
                    stack={item.stack}
                    matchCount={item.matchCount}
                    columns={columns}
                    expanded={expanded}
                    onToggle={() => { onToggleStack(item.stack.number); }}
                  />
                  {expanded && item.rows.map((row) => (row.kind === 'pr'
                    ? <PrRow key={row.pr.number} pr={row.pr} position={row.position} {...rowProps} />
                    : <StackEntryRow key={row.entry.number} entry={row.entry} columns={columns} repo={repo} basePath={basePath} wbIndex={wbIndex} />))}
                </Fragment>
              );
            })}
          </tbody>
        </table>
      </div>
      {listedPrs.some(isBuildOutOfDate) && (
        <p className="mt-2 text-xs text-zinc-500">
          * Built or building against an older commit of the target branch. CI rebuilds it against the latest
          commit when it becomes the merge candidate.
        </p>
      )}
    </>
  );
}

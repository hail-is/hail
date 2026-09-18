import { useState, useRef, useEffect } from 'react';
import type { QueryFieldConfig } from './queryFields';

let rowIdCounter = 0;
function newRowId(): string {
  rowIdCounter += 1;
  return `qb-row-${rowIdCounter}`;
}

interface StructuredRow {
  kind: 'structured';
  id: string;
  field: string;
  operator: string;
  value: string;
}

interface FreeTextRow {
  kind: 'freetext';
  id: string;
  value: string;
}

type Row = StructuredRow | FreeTextRow;

const LINE_PATTERN = /^(\w+)\s*(=~|!~|<=|>=|!=|==|=|<|>)\s*(.+)$/;

function classifyLine(line: string, fields: readonly QueryFieldConfig[]): Row {
  const match = LINE_PATTERN.exec(line.trim());
  if (match) {
    const [, field, operator, value] = match;
    const config = fields.find((f) => f.field === field);
    if (config?.operators.includes(operator)) {
      return { kind: 'structured', id: newRowId(), field, operator, value };
    }
  }
  return { kind: 'freetext', id: newRowId(), value: line.trim() };
}

function parseQuery(q: string, fields: readonly QueryFieldConfig[]): Row[] {
  const lines = q.split('\n').map((l) => l.trim()).filter((l) => l.length > 0);
  return lines.map((line) => classifyLine(line, fields));
}

function buildQuery(rows: Row[]): string {
  return rows
    .map((row) => {
      if (row.kind === 'freetext') return row.value.trim();
      if (row.field && row.operator && row.value) return `${row.field} ${row.operator} ${row.value}`;
      return '';
    })
    .filter((part) => part.length > 0)
    .join('\n');
}

function StructuredRowEditor({ row, fields, onChange, onRemove }: {
  row: StructuredRow;
  fields: readonly QueryFieldConfig[];
  onChange: (_row: StructuredRow) => void;
  onRemove: () => void;
}): JSX.Element {
  const config = fields.find((f) => f.field === row.field);

  return (
    <div className="flex space-x-2 items-center">
      <select
        value={row.field}
        onChange={(e) => { onChange({ ...row, field: e.target.value, operator: '', value: '' }); }}
        className="p-2 bg-white rounded border"
      >
        <option value="">Select a field...</option>
        {fields.map((f) => <option key={f.field} value={f.field}>{f.label}</option>)}
      </select>

      <select
        value={row.operator}
        onChange={(e) => { onChange({ ...row, operator: e.target.value }); }}
        disabled={!config}
        className="p-2 bg-white rounded border"
      >
        <option value="">Select operator...</option>
        {config?.operators.map((op) => <option key={op} value={op}>{op}</option>)}
      </select>

      {config?.valueKind === 'enum' ? (
        <select
          value={row.value}
          onChange={(e) => { onChange({ ...row, value: e.target.value }); }}
          className="p-2 bg-white rounded border flex-grow"
        >
          <option value="">Select value...</option>
          {config.enumValues?.map((v) => <option key={v} value={v}>{v}</option>)}
        </select>
      ) : (
        <input
          type="text"
          value={row.value}
          onChange={(e) => { onChange({ ...row, value: e.target.value }); }}
          placeholder={config?.placeholder ?? 'Enter value...'}
          className="p-2 bg-white rounded border flex-grow"
        />
      )}

      <button
        type="button"
        onClick={onRemove}
        className="text-red-500 hover:text-red-700"
        title="Remove criterion"
      >
        <span className="material-symbols-outlined text-sm">close</span>
      </button>
    </div>
  );
}

function FreeTextRowEditor({ row, onChange, onRemove }: {
  row: FreeTextRow;
  onChange: (_row: FreeTextRow) => void;
  onRemove: () => void;
}): JSX.Element {
  return (
    <div className="flex space-x-2 items-center">
      <span className="text-xs bg-slate-300 rounded px-2 py-1 shrink-0">free text</span>
      <input
        type="text"
        value={row.value}
        onChange={(e) => { onChange({ ...row, value: e.target.value }); }}
        placeholder='partial match &mdash; wrap in "quotes" for exact match'
        className="p-2 bg-white rounded border flex-grow"
      />
      <button
        type="button"
        onClick={onRemove}
        className="text-red-500 hover:text-red-700"
        title="Remove free-text term"
      >
        <span className="material-symbols-outlined text-sm">close</span>
      </button>
    </div>
  );
}

export function QueryBuilder({ fields, q, onSearch, onDirtyChange }: {
  fields: readonly QueryFieldConfig[];
  q: string;
  onSearch: (_q: string) => void;
  onDirtyChange?: (_dirty: boolean) => void;
}): JSX.Element {
  const [mode, setMode] = useState<'builder' | 'text'>('builder');
  const [rows, setRows] = useState<Row[]>(() => parseQuery(q, fields));
  const [textValue, setTextValue] = useState(q);
  const lastSyncedQ = useRef(q);
  const [prevQ, setPrevQ] = useState(q);

  // Reparse synchronously during render (not in a useEffect) so an external `q` change (initial
  // load, status-count filters) paints correctly the first time, with no dirty-flag flash.
  // Skipped when `q` is just the echo of our own submit, so it doesn't clobber later edits.
  if (q !== prevQ) {
    setPrevQ(q);
    if (q !== lastSyncedQ.current) {
      lastSyncedQ.current = q;
      setRows(parseQuery(q, fields));
      setTextValue(q);
    }
  }

  const draftQuery = mode === 'builder' ? buildQuery(rows) : textValue.trim();
  const dirty = draftQuery !== q;

  const onDirtyChangeRef = useRef(onDirtyChange);
  onDirtyChangeRef.current = onDirtyChange;
  useEffect(() => { onDirtyChangeRef.current?.(dirty); }, [dirty]);

  const commit = (newRows: Row[], newText: string) => {
    const built = mode === 'builder' ? buildQuery(newRows) : newText;
    lastSyncedQ.current = built;
    onSearch(built);
  };

  const reset = () => {
    lastSyncedQ.current = q;
    setRows(parseQuery(q, fields));
    setTextValue(q);
  };

  const updateRow = (id: string, updated: Row) => {
    setRows((prev) => prev.map((r) => (r.id === id ? updated : r)));
  };

  const removeRow = (id: string) => {
    setRows((prev) => prev.filter((r) => r.id !== id));
  };

  const addStructuredRow = () => {
    setRows((prev) => [...prev, { kind: 'structured', id: newRowId(), field: '', operator: '', value: '' }]);
  };

  const addFreeTextRow = () => {
    setRows((prev) => [...prev, { kind: 'freetext', id: newRowId(), value: '' }]);
  };

  const switchToText = () => {
    if (mode === 'text') return;
    setTextValue(buildQuery(rows));
    setMode('text');
  };

  const switchToBuilder = () => {
    if (mode === 'builder') return;
    setRows(parseQuery(textValue, fields));
    setMode('builder');
  };

  const actionRow = (
    <div className="flex justify-end items-center gap-2">
      {dirty && (
        <span className="flex items-center gap-1 text-sm text-yellow-700">
          <span className="material-symbols-outlined text-base leading-none">warning</span>
          Filters changed, search to apply
        </span>
      )}
      {dirty && (
        <button
          type="button"
          onClick={reset}
          className="px-3 py-1 bg-white rounded border text-sm hover:bg-slate-50"
        >
          Reset
        </button>
      )}
      <button
        type="submit"
        disabled={!dirty}
        className="px-3 py-1 bg-blue-500 text-white rounded text-sm hover:bg-blue-600 disabled:opacity-50 disabled:hover:bg-blue-500"
      >
        Search
      </button>
    </div>
  );

  return (
    <form
      className="flex space-x-4"
      onSubmit={(e) => { e.preventDefault(); commit(rows, textValue); }}
    >
      <div className="flex-grow">
        <div className="flex space-x-1">
          <button
            type="button"
            onClick={switchToBuilder}
            className={`px-3 py-1 text-sm rounded-t ${
              mode === 'builder' ? 'bg-slate-200 font-medium' : 'bg-slate-100 text-zinc-500 hover:bg-slate-200'
            }`}
          >
            Query Builder
          </button>
          <button
            type="button"
            onClick={switchToText}
            className={`px-3 py-1 text-sm rounded-t ${
              mode === 'text' ? 'bg-slate-200 font-medium' : 'bg-slate-100 text-zinc-500 hover:bg-slate-200'
            }`}
          >
            Text Query
          </button>
        </div>

        {mode === 'builder' ? (
          <div className="min-h-24 w-full p-4 bg-slate-200 rounded-b rounded-tr space-y-3">
            {rows.length === 0 && (
              <div className="p-2 bg-white rounded border text-sm text-zinc-500">Showing all jobs</div>
            )}
            {rows.map((row) => (row.kind === 'structured' ? (
              <StructuredRowEditor
                key={row.id}
                row={row}
                fields={fields}
                onChange={(r) => { updateRow(row.id, r); }}
                onRemove={() => { removeRow(row.id); }}
              />
            ) : (
              <FreeTextRowEditor
                key={row.id}
                row={row}
                onChange={(r) => { updateRow(row.id, r); }}
                onRemove={() => { removeRow(row.id); }}
              />
            )))}

            <div className="flex justify-between items-center flex-wrap gap-2">
              <div className="flex items-center gap-2 text-sm">
                <span className="text-zinc-500">Add:</span>
                <button
                  type="button"
                  onClick={addStructuredRow}
                  className="px-3 py-1 bg-blue-500 text-white rounded text-sm hover:bg-blue-600"
                >
                  Search Criterion
                </button>
                <button
                  type="button"
                  onClick={addFreeTextRow}
                  className="px-3 py-1 bg-blue-500 text-white rounded text-sm hover:bg-blue-600"
                >
                  Free text filter
                </button>
              </div>

              {actionRow}
            </div>
          </div>
        ) : (
          <div className="min-h-32 w-full p-4 bg-slate-200 rounded-b rounded-tr space-y-3">
            <textarea
              value={textValue}
              onChange={(e) => { setTextValue(e.target.value); }}
              placeholder='Enter search query (e.g., "cost > 5.00")'
              spellCheck={false}
              autoCorrect="off"
              className="h-32 w-full p-4 bg-white rounded resize whitespace-pre border"
            />

            {actionRow}
          </div>
        )}
      </div>

      <div className="flex items-start">
        <a href="https://hail.is/docs/batch/advanced_search_help.html" target="_blank" rel="noreferrer">
          <span className="material-symbols-outlined">help</span>
        </a>
      </div>
    </form>
  );
}

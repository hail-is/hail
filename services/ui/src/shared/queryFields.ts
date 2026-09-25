export type QueryFieldValueKind = 'string' | 'number' | 'datetime' | 'enum';

export interface QueryFieldConfig {
  readonly field: string;
  readonly label: string;
  readonly operators: readonly string[];
  readonly valueKind: QueryFieldValueKind;
  readonly placeholder?: string;
  readonly enumValues?: readonly string[];
}

const STRING_OPS = ['=', '!=', '=~', '!~'] as const;
const EXACT_OPS = ['=', '!='] as const;
const NUMERIC_OPS = ['=', '!=', '>', '>=', '<', '<='] as const;

const sharedFields: readonly QueryFieldConfig[] = [
  { field: 'name', label: 'Name', operators: STRING_OPS, valueKind: 'string', placeholder: 'Enter name...' },
  { field: 'cost', label: 'Cost', operators: NUMERIC_OPS, valueKind: 'number', placeholder: 'Enter cost (e.g., 5.00)...' },
  { field: 'duration', label: 'Duration', operators: NUMERIC_OPS, valueKind: 'number', placeholder: 'Enter duration in seconds...' },
  {
    field: 'start_time', label: 'Start Time', operators: NUMERIC_OPS, valueKind: 'datetime',
    placeholder: 'Enter date (ISO format, e.g., 2025-02-27T17:15:25Z)...',
  },
  {
    field: 'end_time', label: 'End Time', operators: NUMERIC_OPS, valueKind: 'datetime',
    placeholder: 'Enter date (ISO format, e.g., 2025-02-27T17:15:25Z)...',
  },
];

const BATCH_STATE_VALUES = ['running', 'complete', 'success', 'failure', 'cancelled', 'open', 'closed'] as const;
const JOB_STATE_VALUES = [
  'pending', 'ready', 'creating', 'running', 'live', 'cancelled', 'error', 'failed', 'bad', 'success', 'done',
] as const;

function stateField(enumValues: readonly string[]): QueryFieldConfig {
  return { field: 'state', label: 'State', operators: EXACT_OPS, valueKind: 'enum', enumValues };
}

const batchOnlyFields: readonly QueryFieldConfig[] = [
  stateField(BATCH_STATE_VALUES),
  { field: 'batch_id', label: 'Batch ID', operators: NUMERIC_OPS, valueKind: 'number', placeholder: 'Enter batch ID...' },
  { field: 'user', label: 'User', operators: EXACT_OPS, valueKind: 'string', placeholder: 'Enter username...' },
  { field: 'billing_project', label: 'Billing Project', operators: EXACT_OPS, valueKind: 'string', placeholder: 'Enter billing project...' },
];

const jobOnlyFields: readonly QueryFieldConfig[] = [
  { field: 'job_id', label: 'Job ID', operators: NUMERIC_OPS, valueKind: 'number', placeholder: 'Enter job ID...' },
  stateField(JOB_STATE_VALUES),
  { field: 'instance', label: 'Instance', operators: STRING_OPS, valueKind: 'string', placeholder: 'Enter instance name...' },
  {
    field: 'instance_collection', label: 'Instance Collection', operators: STRING_OPS, valueKind: 'string',
    placeholder: 'Enter instance collection...',
  },
  { field: 'exit_code', label: 'Exit Code', operators: NUMERIC_OPS, valueKind: 'number', placeholder: 'Enter exit code (e.g., 0, 1)...' },
];

export const batchQueryFields: readonly QueryFieldConfig[] = [...sharedFields, ...batchOnlyFields];
export const jobQueryFields: readonly QueryFieldConfig[] = [...sharedFields, ...jobOnlyFields];

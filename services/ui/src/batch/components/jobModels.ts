export interface TimingEntry {
  start_time?: number | null;
  finish_time?: number | null;
  duration?: number | null;
}

export type ContainerTiming = {
  [key: string]: TimingEntry | null | undefined;
};

export interface ContainerStatus {
  name: string;
  state: string;
  short_error?: string | null;
  error?: string | null;
  timing: ContainerTiming;
}

export type JobStatus = {
  container_statuses?: {
    input?: ContainerStatus | null;
    main?: ContainerStatus | null;
    output?: ContainerStatus | null;
  };
  error?: string | null;
};

export interface JobSpec {
  process?: { type: 'docker' | 'jvm'; image?: string; command?: string[] };
  user_code?: string;
  resources?: Record<string, unknown>;
  env?: { name: string; value: string }[];
  input_files?: [string, string][];
  output_files?: [string, string][];
  always_run?: boolean;
  n_max_attempts?: number;
  network?: string;
  regions?: string[];
}

export type Job = {
  id: number;
  batch_id: number;
  state: string;
  exit_code?: number | null;
  duration?: string;
  cost?: number;
  cost_breakdown?: { resource: string; cost: number }[] | null;
  user?: string;
  billing_project?: string;
  always_run?: boolean;
  attributes?: Record<string, string>;
  inst_coll?: string;
  spec?: JobSpec | null;
  status?: JobStatus | null;
};

export type Attempt = {
  attempt_id: string;
  instance_name?: string;
  start_time?: string;
  start_time_ms?: number;
  end_time?: string;
  end_time_ms?: number;
  duration?: string;
  duration_ms?: number;
  reason?: string;
};

export const TERMINAL_STATES = new Set(['Success', 'Failed', 'Error', 'Cancelled']);

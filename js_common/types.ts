export type Maybe<T> = T | undefined;

export type Batch = {
  id: number,
  user: string,
  billing_project: string,
  token: string,
  state: string,
  complete: boolean,
  closed: boolean,
  n_jobs: number,
  n_completed: number,
  n_succeeded: number,
  n_failed: number,
  n_cancelled: number,
  time_created: string,
  time_closed: string,
  time_completed: string,
  duration: string,
  attributes: any,
  msec_mcpu: number,
  cost: string,
}

export type Job = {
  batch_id: number,
  billing_project: string,
  cost: number,
  duration: number,
  exit_code: Maybe<number>,
  job_id: number,
  msec_mcpu: number,
  name: Maybe<string>,
  state: string,
  user: string,
}

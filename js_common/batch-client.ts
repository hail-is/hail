import type { Job, Batch } from './types'

export type GetJobsResult = { jobs: Job[] }
export type GetBatchesResult = {
  batches: Batch[],
  last_batch_id: number,
}

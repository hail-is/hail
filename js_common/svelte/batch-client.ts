import { pollingApiStore } from './store'
import type { StoreApiResult } from './store'
import { GetJobsResult, GetBatchesResult } from '../batch-client'

export function getJobsStore(id: number): StoreApiResult<GetJobsResult> {
  return pollingApiStore(`/api/v1alpha/batches/${id}/jobs`)
}

export function getBatchesStore(): StoreApiResult<GetBatchesResult> {
  return pollingApiStore('/api/v1alpha/batches')
}

import { usePollingApi } from './hooks'
import { Maybe } from '../types'
import { GetJobsResult, GetBatchesResult } from '../batch-client'

export function useJobs(id: number): Maybe<GetJobsResult> {
  return usePollingApi(`/api/v1alpha/batches/${id}/jobs`)
}

export function useBatches(): Maybe<GetBatchesResult> {
  return usePollingApi('/api/v1alpha/batches')
}

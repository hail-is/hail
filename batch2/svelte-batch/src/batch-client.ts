import type { GetJobsResult, GetBatchesResult } from '../../../js_common/src/batch-client'

export function getJobsStore(id: number): Writable<Maybe<GetJobsResult>> {
  return streamingApiStore<GetJobsResult>(`/api/v1alpha/batches/${id}/jobs`);
}

export function getBatchesStore(): Writable<Maybe<GetBatchesResult>> {
  return streamingApiStore<GetBatchesResult>('/api/v1alpha/batches');
}

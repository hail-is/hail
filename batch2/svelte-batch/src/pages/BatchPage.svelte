<script lang='ts'>
  import { onDestroy } from 'svelte'
  import { getJobsStore } from '@hail/common/svelte/batch-client'
  import JobTable from '../components/JobTable.svelte'
  import type { Maybe, Job } from '@hail/common/types'

  export let id: number;

  let jobs: Maybe<Job[]> = undefined;
  const { store, destroy } = getJobsStore(id)
  const unsubscribe = store.subscribe(msg => {
    jobs = msg?.jobs
  })

  onDestroy(() => {
    unsubscribe()
    destroy()
  })
</script>

{#if jobs}
  <JobTable jobs={jobs} batchId={id} />
{:else}
  <div>Loading...</div>
{/if}

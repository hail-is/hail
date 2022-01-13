<script lang='ts'>
  import { onDestroy } from 'svelte'
  import { getBatchesStore } from '@hail/common/svelte/batch-client'
  import type { Maybe, Batch } from '@hail/common/types'
  import BatchTable from '../components/BatchTable.svelte'

  let batches: Maybe<Batch[]> = undefined
  let { store, destroy } = getBatchesStore()

  const unsubscribe = store.subscribe(msg => {
    batches = msg?.batches
  })

  onDestroy(() => {
    unsubscribe()
    destroy()
  })
</script>

{#if batches}
  <BatchTable batches={batches} />
{:else}
  <div>Loading...</div>
{/if}

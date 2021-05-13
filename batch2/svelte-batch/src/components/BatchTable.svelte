<script lang='ts'>
  import { Link } from 'svelte-navigator'
  import type { Batch } from '@hail/common/types'
  export let batches: Batch[]
</script>

<h1>Batches</h1>
<ul>
  {#each batches as batch}
    <li>
      <Link to={`/batches/${batch.id}`}>{batch.id}</Link>
      <span>{batch.state}</span>
    </li>
  {/each}
</ul>

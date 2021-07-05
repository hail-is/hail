<script lang='ts'>
  import type { Job } from '@hail/common/types';
  import '@hail/common/hail.css';

  export let jobs: Job[];
  export let batchId: number;
</script>

<h1>Batch #{batchId}</h1>
<ol>
  {#each jobs as job}
    <li class='List'>{JSON.stringify(job, null, 2)}</li>
  {/each}
</ol>

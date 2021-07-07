<script lang="ts">
  import { Router, Route } from 'svelte-navigator'
  import '@hail/common/hail.css'
  import BatchesPage from './pages/BatchesPage.svelte'
  import BatchPage from './pages/BatchPage.svelte'
</script>

<main>
  <Router>
    <header class="App">
      <Route exact path="/batches/:id" let:params>
        <BatchPage id={parseInt(params.id)} />
      </Route>
      <Route exact path="/">
        <BatchesPage />
      </Route>
      <Route exact path="*">
        <p>Uh oh! That page does not exist...</p>
      </Route>
    </header>
  </Router>
</main>

import React from 'react';
import { useBatches } from '@hail/common/react/batch-client';
import BatchTable from '../components/BatchTable';

export default function BatchesPage() {
  const batches = useBatches();

  return (
    <>
      <h1>Batches</h1>
      {batches ? <BatchTable batches={batches.batches} /> : <div>Loading...</div>}
    </>
  );
}

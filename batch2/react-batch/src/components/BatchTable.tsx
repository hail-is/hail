import React from 'react';
import { Link } from 'react-router-dom';
import type { Batch } from '@hail/common/types';

function BatchTableRow({ batch }: { batch: Batch }) {
  return (
    <>
      <Link to={`/batches/${batch.id}`}>{batch.id}</Link>
      <span>{batch.state}</span>
    </>
  );
}

export default function BatchTable({ batches }: { batches: Batch[] }) {
  return (
    <ul>
      {batches.map((b) => <li key={b.id}><BatchTableRow batch={b} /></li>)}
    </ul>
  );
}

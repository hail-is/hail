import React from 'react';
import type { Job } from '@hail/common/types';

function JobTableRow({ job }: { job: Job }) {
  return <div>{JSON.stringify(job, null, 2)}</div>;
}

type JobTableProps = {
  batchId: number,
  jobs: Job[],
}
export default function JobTable({ batchId, jobs }: JobTableProps) {
  return (
    <>
      <h1>Batch #{batchId}</h1>
      <ol>
        {jobs.map((j) => <li className="List" key={j.job_id}><JobTableRow job={j} /></li>)}
      </ol>
    </>
  );
}

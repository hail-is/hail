import React from 'react';
import { useParams } from 'react-router-dom';
import { useJobs } from '@hail/common/react/batch-client';
import JobTable from '../components/JobTable';

import '@hail/common/hail.css';

type BatchPageParams = { id?: string };
export default function BatchPage() {
  const id = parseInt(useParams<BatchPageParams>().id!, 10);
  const jobs = useJobs(id);

  return jobs ? <JobTable batchId={id} jobs={jobs.jobs} /> : <div>Loading...</div>;
}

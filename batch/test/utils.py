from hailtop.batch_client.client import Job


def batch_status_job_counter(batch_status, job_state):
    return len([j for j in batch_status['jobs'] if j['state'] == job_state])


def legacy_batch_status(batch):
    status = batch.status()
    status['jobs'] = list(batch.jobs())
    return status

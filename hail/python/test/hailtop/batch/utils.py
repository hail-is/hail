from hailtop.batch_client.client import Job


def job_logs(batch):
    logs = {}
    jobs = batch.jobs()
    for j_status in jobs:
        j = Job(batch, j_status['job_id'])
        logs[j.id] = j.log()
    return logs

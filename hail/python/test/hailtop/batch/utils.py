import hailtop.batch_client.client as bc


def debug_info(batch: bc.Batch):
    jobs = batch.jobs()
    for j_status in jobs:
        j_status['log'] = batch.get_job_log(j_status['job_id'])
    return str(jobs)

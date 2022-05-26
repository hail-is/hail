
INSERT INTO inst_colls (`name`, `is_pool`, `boot_disk_size_gb`, `max_instances`, `max_live_instances`, `cloud`)
SELECT 'seqr-standard', 1, boot_disk_size_gb, max_instances, max_live_instances, cloud
FROM inst_colls
WHERE name = 'standard';

INSERT INTO inst_colls (`name`, `is_pool`, `boot_disk_size_gb`, `max_instances`, `max_live_instances`, `cloud`)
SELECT 'seqr-highmem', 1, boot_disk_size_gb, max_instances, max_live_instances, cloud
FROM inst_colls
WHERE name = 'highmem';

INSERT INTO inst_colls (`name`, `is_pool`, `boot_disk_size_gb`, `max_instances`, `max_live_instances`, `cloud`)
SELECT 'seqr-highcpu', 1, boot_disk_size_gb, max_instances, max_live_instances, cloud
FROM inst_colls
WHERE name = 'highcpu';

INSERT INTO pools (`name`, `worker_type`, `worker_cores`, `worker_local_ssd_data_disk`,
  `worker_external_ssd_data_disk_size_gb`, `enable_standing_worker`, `standing_worker_cores`,
  `preemptible`)
SELECT 'seqr-standard', worker_type, worker_cores, worker_local_ssd_data_disk,
  worker_external_ssd_data_disk_size_gb, FALSE, standing_worker_cores,
  TRUE
FROM pools
WHERE name = 'standard';

INSERT INTO pools (`name`, `worker_type`, `worker_cores`, `worker_local_ssd_data_disk`,
  `worker_external_ssd_data_disk_size_gb`, `enable_standing_worker`, `standing_worker_cores`,
  `preemptible`)
SELECT 'seqr-highmem', worker_type, worker_cores, worker_local_ssd_data_disk,
  worker_external_ssd_data_disk_size_gb, FALSE, standing_worker_cores,
  TRUE
FROM pools
WHERE name = 'highmem';

INSERT INTO pools (`name`, `worker_type`, `worker_cores`, `worker_local_ssd_data_disk`,
  `worker_external_ssd_data_disk_size_gb`, `enable_standing_worker`, `standing_worker_cores`,
  `preemptible`)
SELECT 'seqr-highcpu', worker_type, worker_cores, worker_local_ssd_data_disk,
  worker_external_ssd_data_disk_size_gb, FALSE, standing_worker_cores,
  TRUE
FROM pools
WHERE name = 'highcpu';

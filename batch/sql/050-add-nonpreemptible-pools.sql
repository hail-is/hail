
INSERT INTO inst_colls (`name`, `is_pool`, `boot_disk_size_gb`, `max_instances`, `max_live_instances`, `cloud`)
SELECT 'standard-np', 1, boot_disk_size_gb, max_instances, max_live_instances, cloud
FROM inst_colls
WHERE name = 'standard';

INSERT INTO inst_colls (`name`, `is_pool`, `boot_disk_size_gb`, `max_instances`, `max_live_instances`, `cloud`)
SELECT 'highmem-np', 1, boot_disk_size_gb, max_instances, max_live_instances, cloud
FROM inst_colls
WHERE name = 'highmem';

INSERT INTO inst_colls (`name`, `is_pool`, `boot_disk_size_gb`, `max_instances`, `max_live_instances`, `cloud`)
SELECT 'highcpu-np', 1, boot_disk_size_gb, max_instances, max_live_instances, cloud
FROM inst_colls
WHERE name = 'highcpu';

ALTER TABLE `pools` ADD `preemptible` BOOLEAN NOT NULL DEFAULT TRUE;

INSERT INTO pools (`name`, `worker_type`, `worker_cores`, `worker_local_ssd_data_disk`,
  `worker_external_ssd_data_disk_size_gb`, `enable_standing_worker`, `standing_worker_cores`,
  `preemptible`)
SELECT 'standard-np', worker_type, worker_cores, worker_local_ssd_data_disk,
  worker_external_ssd_data_disk_size_gb, enable_standing_worker, standing_worker_cores,
  FALSE
FROM pools
WHERE name = 'standard';

INSERT INTO pools (`name`, `worker_type`, `worker_cores`, `worker_local_ssd_data_disk`,
  `worker_external_ssd_data_disk_size_gb`, `enable_standing_worker`, `standing_worker_cores`,
  `preemptible`)
SELECT 'highmem-np', worker_type, worker_cores, worker_local_ssd_data_disk,
  worker_external_ssd_data_disk_size_gb, enable_standing_worker, standing_worker_cores,
  FALSE
FROM pools
WHERE name = 'highmem';

INSERT INTO pools (`name`, `worker_type`, `worker_cores`, `worker_local_ssd_data_disk`,
  `worker_external_ssd_data_disk_size_gb`, `enable_standing_worker`, `standing_worker_cores`,
  `preemptible`)
SELECT 'highcpu-np', worker_type, worker_cores, worker_local_ssd_data_disk,
  worker_external_ssd_data_disk_size_gb, enable_standing_worker, standing_worker_cores,
  FALSE
FROM pools
WHERE name = 'highcpu';

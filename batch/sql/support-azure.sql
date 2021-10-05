ALTER TABLE inst_colls MODIFY COLUMN `boot_disk_size_gb` BIGINT;
ALTER TABLE pools CHANGE COLUMN `worker_local_ssd_data_disk` `local_ssd_data_disk` BOOLEAN NOT NULL DEFAULT 1;
ALTER TABLE pools CHANGE COLUMN `worker_pd_ssd_data_disk_size_gb` `external_data_disk_size_gb` BIGINT NOT NULL DEFAULT 0;

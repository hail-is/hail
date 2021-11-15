ALTER TABLE pools CHANGE COLUMN `worker_pd_ssd_data_disk_size_gb` `worker_external_ssd_data_disk_size_gb` BIGINT NOT NULL DEFAULT 0;

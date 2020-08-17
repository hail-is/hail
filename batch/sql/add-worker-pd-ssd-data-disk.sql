ALTER TABLE `globals` ADD COLUMN `worker_local_ssd_data_disk` BOOLEAN NOT NULL DEFAULT 1;
ALTER TABLE `globals` ADD COLUMN `worker_pd_ssd_data_disk_size_gb` BIGINT NOT NULL DEFAULT 0;

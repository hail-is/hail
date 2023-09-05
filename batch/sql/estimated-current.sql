CREATE TABLE IF NOT EXISTS `globals` (
  `instance_id` VARCHAR(100) NOT NULL,
  `internal_token` VARCHAR(100) NOT NULL,  # deprecated
  `n_tokens` INT NOT NULL,
  `frozen` BOOLEAN NOT NULL DEFAULT FALSE
) ENGINE = InnoDB;

CREATE TABLE IF NOT EXISTS `feature_flags` (
  `compact_billing_tables` BOOLEAN NOT NULL,
  `oms_agent` BOOLEAN NOT NULL DEFAULT TRUE
) ENGINE = InnoDB;

CREATE TABLE IF NOT EXISTS `resources` (
  `resource` VARCHAR(100) NOT NULL,
  `rate` DOUBLE NOT NULL,
  `resource_id` INT AUTO_INCREMENT UNIQUE NOT NULL,
  `deduped_resource_id` INT DEFAULT NULL,
  PRIMARY KEY (`resource`)
) ENGINE = InnoDB;
CREATE INDEX `resources_deduped_resource_id` ON `resources` (`deduped_resource_id`);

CREATE TABLE IF NOT EXISTS `latest_product_versions` (
  `product` VARCHAR(100) NOT NULL,
  `version` VARCHAR(100) NOT NULL,
  `sku` VARCHAR(100),
  `time_updated` TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`product`)
) ENGINE = InnoDB;

CREATE TABLE IF NOT EXISTS `inst_colls` (
  `name` VARCHAR(255) NOT NULL,
  `is_pool` BOOLEAN NOT NULL,
  `boot_disk_size_gb` BIGINT NOT NULL,
  `max_instances` BIGINT NOT NULL,
  `max_live_instances` BIGINT NOT NULL,
  `cloud` VARCHAR(100) NOT NULL,
  `max_new_instances_per_autoscaler_loop` INT NOT NULL,
  `autoscaler_loop_period_secs` INT NOT NULL,
  `worker_max_idle_time_secs` INT NOT NULL,
  PRIMARY KEY (`name`)
) ENGINE = InnoDB;
CREATE INDEX `inst_colls_is_pool` ON `inst_colls` (`is_pool`);

INSERT INTO inst_colls (`name`, `is_pool`, `boot_disk_size_gb`, `max_instances`, `max_live_instances`) VALUES ('standard', 1, 10, 6250, 800);
INSERT INTO inst_colls (`name`, `is_pool`, `boot_disk_size_gb`, `max_instances`, `max_live_instances`) VALUES ('highmem', 1, 10, 6250, 800);
INSERT INTO inst_colls (`name`, `is_pool`, `boot_disk_size_gb`, `max_instances`, `max_live_instances`) VALUES ('highcpu', 1, 10, 6250, 800);
INSERT INTO inst_colls (`name`, `is_pool`, `boot_disk_size_gb`, `max_instances`, `max_live_instances`) VALUES ('job-private', 0, 10, 6250, 800);

CREATE TABLE IF NOT EXISTS `pools` (
  `name` VARCHAR(255) NOT NULL,
  `worker_type` VARCHAR(100) NOT NULL,
  `worker_cores` BIGINT NOT NULL,
  `worker_local_ssd_data_disk` BOOLEAN NOT NULL DEFAULT 1,
  `worker_external_ssd_data_disk_size_gb` BIGINT NOT NULL DEFAULT 0,
  `enable_standing_worker` BOOLEAN NOT NULL DEFAULT FALSE,
  `standing_worker_cores` BIGINT NOT NULL DEFAULT 0,
  `preemptible` BOOLEAN NOT NULL DEFAULT TRUE,
  `standing_worker_max_idle_time_secs` INT NOT NULL,
  `job_queue_scheduling_window_secs` INT NOT NULL,
  `min_instances` BIGINT NOT NULL,
  PRIMARY KEY (`name`),
  FOREIGN KEY (`name`) REFERENCES inst_colls(name) ON DELETE CASCADE
) ENGINE = InnoDB;

INSERT INTO pools (`name`, `worker_type`, `worker_cores`, `worker_local_ssd_data_disk`,
  `worker_external_ssd_data_disk_size_gb`, `enable_standing_worker`, `standing_worker_cores`)
VALUES ('standard', 'standard', 16, 1, 0, 1, 4);

INSERT INTO pools (`name`, `worker_type`, `worker_cores`, `worker_local_ssd_data_disk`,
  `worker_external_ssd_data_disk_size_gb`, `enable_standing_worker`, `standing_worker_cores`)
VALUES ('highmem', 'highmem', 16, 10, 1, 0, 0, 4);

INSERT INTO pools (`name`, `worker_type`, `worker_cores`, `worker_local_ssd_data_disk`,
  `worker_external_ssd_data_disk_size_gb`, `enable_standing_worker`, `standing_worker_cores`)
VALUES ('highcpu', 'highcpu', 16, 10, 1, 0, 0, 4);

CREATE TABLE IF NOT EXISTS `billing_projects` (
  `name` VARCHAR(100) NOT NULL,
  `name_cs` VARCHAR(100) NOT NULL COLLATE utf8mb4_0900_as_cs,
  `status` ENUM('open', 'closed', 'deleted') NOT NULL DEFAULT 'open',
  `limit` DOUBLE DEFAULT NULL,
  `msec_mcpu` BIGINT DEFAULT 0,
  PRIMARY KEY (`name`)
) ENGINE = InnoDB;
CREATE INDEX `billing_project_status` ON `billing_projects` (`status`);
CREATE UNIQUE INDEX `billing_project_name_cs` ON `billing_projects` (`name_cs`);
CREATE INDEX `billing_project_name_cs_status` ON `billing_projects` (`name_cs`, `status`);

CREATE TABLE IF NOT EXISTS `billing_project_users` (
  `billing_project` VARCHAR(100) NOT NULL,
  `user` VARCHAR(100) NOT NULL,
  `user_cs` VARCHAR(100) NOT NULL COLLATE utf8mb4_0900_as_cs,
  PRIMARY KEY (`billing_project`, `user`),
  FOREIGN KEY (`billing_project`) REFERENCES billing_projects(name) ON DELETE CASCADE
) ENGINE = InnoDB;
CREATE INDEX `billing_project_users_billing_project_user_cs` ON `billing_project_users` (`billing_project`, `user_cs`);

INSERT INTO `billing_projects` (`name`)
VALUES ('ci');

INSERT INTO `billing_projects` (`name`)
VALUES ('test');

INSERT INTO `billing_project_users` (`billing_project`, `user`)
VALUES ('ci', 'ci');

INSERT INTO `billing_project_users` (`billing_project`, `user`)
VALUES ('test', 'test');

INSERT INTO `billing_project_users` (`billing_project`, `user`)
VALUES ('test', 'test-dev');

CREATE TABLE IF NOT EXISTS `instances` (
  `name` VARCHAR(100) NOT NULL,
  `state` VARCHAR(40) NOT NULL,
  `activation_token` VARCHAR(100),
  `token` VARCHAR(100) NOT NULL,
  `cores_mcpu` INT NOT NULL,
  `time_created` BIGINT NOT NULL,
  `failed_request_count` INT NOT NULL DEFAULT 0,
  `last_updated` BIGINT NOT NULL,
  `ip_address` VARCHAR(100),
  `time_activated` BIGINT,
  `time_deactivated` BIGINT,
  `removed` BOOLEAN NOT NULL DEFAULT FALSE,
  `version` INT NOT NULL,
  `location` VARCHAR(40) NOT NULL,
  `inst_coll` VARCHAR(255) NOT NULL,
  `machine_type` VARCHAR(255) NOT NULL,
  `preemptible` BOOLEAN NOT NULL,
  `instance_config` MEDIUMTEXT,
  PRIMARY KEY (`name`),
  FOREIGN KEY (`inst_coll`) REFERENCES inst_colls(`name`)
) ENGINE = InnoDB;
CREATE INDEX `instances_removed` ON `instances` (`removed`);
CREATE INDEX `instances_inst_coll` ON `instances` (`inst_coll`);
CREATE INDEX `instances_removed_inst_coll` ON `instances` (`removed`, `inst_coll`);
CREATE INDEX `instances_time_activated` ON `instances` (`time_activated`);

CREATE TABLE IF NOT EXISTS `instances_free_cores_mcpu` (
  `name` VARCHAR(100) NOT NULL,
  `free_cores_mcpu` INT NOT NULL,
  PRIMARY KEY (`name`),
  FOREIGN KEY (`name`) REFERENCES instances(`name`) ON DELETE CASCADE
) ENGINE = InnoDB;

CREATE TABLE IF NOT EXISTS `user_inst_coll_resources` (
  `user` VARCHAR(100) NOT NULL,
  `inst_coll` VARCHAR(255),
  `token` INT NOT NULL,
  `n_ready_jobs` INT NOT NULL DEFAULT 0,
  `n_running_jobs` INT NOT NULL DEFAULT 0,
  `n_creating_jobs` INT NOT NULL DEFAULT 0,
  `ready_cores_mcpu` BIGINT NOT NULL DEFAULT 0,
  `running_cores_mcpu` BIGINT NOT NULL DEFAULT 0,
  `n_cancelled_ready_jobs` INT NOT NULL DEFAULT 0,
  `n_cancelled_running_jobs` INT NOT NULL DEFAULT 0,
  `n_cancelled_creating_jobs` INT NOT NULL DEFAULT 0,
  PRIMARY KEY (`user`, `inst_coll`, `token`),
  FOREIGN KEY (`inst_coll`) REFERENCES inst_colls(name) ON DELETE CASCADE
) ENGINE = InnoDB;
CREATE INDEX `user_inst_coll_resources_inst_coll` ON `user_inst_coll_resources` (`inst_coll`);

CREATE TABLE IF NOT EXISTS `batches` (
  `id` BIGINT NOT NULL AUTO_INCREMENT,
  `userdata` TEXT NOT NULL,
  `user` VARCHAR(100) NOT NULL,
  `billing_project` VARCHAR(100) NOT NULL,
  `attributes` TEXT,
  `callback` TEXT,
  `state` VARCHAR(40) NOT NULL,
  `deleted` BOOLEAN NOT NULL DEFAULT FALSE,
  `n_jobs` INT NOT NULL,
  `time_created` BIGINT NOT NULL,
  `time_closed` BIGINT,
  `time_completed` BIGINT,
  `msec_mcpu` BIGINT NOT NULL DEFAULT 0,
  `token` VARCHAR(100) DEFAULT NULL,
  `format_version` INT NOT NULL,
  `cancel_after_n_failures` INT DEFAULT NULL,
  `migrated_batch` BOOLEAN NOT NULL DEFAULT 0,
  PRIMARY KEY (`id`),
  FOREIGN KEY (`billing_project`) REFERENCES billing_projects(name)
) ENGINE = InnoDB;
CREATE INDEX `batches_state` ON `batches` (`state`);
CREATE INDEX `batches_user_state` ON `batches` (`user`, `state`);
CREATE INDEX `batches_deleted` ON `batches` (`deleted`);
CREATE INDEX `batches_token` ON `batches` (`token`);
CREATE INDEX `batches_time_completed` ON `batches` (`time_completed`);
CREATE INDEX `batches_billing_project_state` ON `batches` (`billing_project`, `state`);

DROP TABLE IF EXISTS `job_groups`;
CREATE TABLE IF NOT EXISTS `job_groups` (
  `batch_id` BIGINT NOT NULL,
  `job_group_id` INT NOT NULL,
  `user` VARCHAR(100) NOT NULL,
  `attributes` TEXT,
  `cancel_after_n_failures` INT DEFAULT NULL,
  `state` ENUM('running', 'complete') NOT NULL,
  `n_jobs` INT NOT NULL,
  `time_created` BIGINT NOT NULL,
  `time_completed` BIGINT,
  `callback` VARCHAR(255),
  PRIMARY KEY (`batch_id`, `job_group_id`),
  FOREIGN KEY (`batch_id`) REFERENCES batches(`id`) ON DELETE CASCADE
) ENGINE = InnoDB;
CREATE INDEX `job_groups_user_state` ON `job_groups` (`user`, `state`);  # used to get cancelled job groups by user
CREATE INDEX `job_groups_state_callback` ON `job_groups` (`batch_id`, `state`, `callback`);  # used in callback on job group completion
CREATE INDEX `job_groups_time_created` ON `job_groups` (`batch_id`, `time_created`);  # used in list job groups and UI
CREATE INDEX `job_groups_time_completed` ON `job_groups` (`batch_id`, `time_completed`);  # used in list job groups and UI
CREATE INDEX `job_groups_state_cancel_after_n_failures` ON `job_groups` (`state`, `cancel_after_n_failures`);  # used in cancelling any cancel fast job groups

DROP TABLE IF EXISTS `job_group_parents`;
CREATE TABLE IF NOT EXISTS `job_group_parents` (
  `batch_id` BIGINT NOT NULL,
  `job_group_id` INT NOT NULL,
  `parent_id` INT NOT NULL,
  `level` INT NOT NULL,
  PRIMARY KEY (`batch_id`, `job_group_id`, `parent_id`),
  FOREIGN KEY (`batch_id`, `job_group_id`) REFERENCES job_groups (`batch_id`, `job_group_id`) ON DELETE CASCADE,
  FOREIGN KEY (`batch_id`, `parent_id`) REFERENCES job_groups (`batch_id`, `job_group_id`) ON DELETE CASCADE
) ENGINE = InnoDB;
CREATE INDEX `job_group_parents_parent_id_level` ON `job_group_parents` (`batch_id`, `parent_id`, `level`);
CREATE INDEX `job_group_parents_job_group_id_level` ON `job_group_parents` (`batch_id`, `job_group_id`, `level`);

CREATE TABLE IF NOT EXISTS `batch_updates` (
  `batch_id` BIGINT NOT NULL,
  `update_id` INT NOT NULL,
  `token` VARCHAR(100) DEFAULT NULL,
  `start_job_id` INT NOT NULL,
  `n_jobs` INT NOT NULL,
  `committed` BOOLEAN NOT NULL DEFAULT FALSE,
  `time_created` BIGINT NOT NULL,
  `time_committed` BIGINT,
  PRIMARY KEY (`batch_id`, `update_id`),
  FOREIGN KEY (`batch_id`) REFERENCES batches(`id`),
  UNIQUE KEY (`batch_id`, `start_job_id`)
) ENGINE = InnoDB;
CREATE INDEX `batch_updates_committed` ON `batch_updates` (`batch_id`, `committed`);
CREATE INDEX `batch_updates_start_job_id` ON `batch_updates` (`batch_id`, `start_job_id`);

CREATE TABLE IF NOT EXISTS `batches_n_jobs_in_complete_states` (
  `id` BIGINT NOT NULL,
  `job_group_id` INT NOT NULL DEFAULT 0,
  `n_completed` INT NOT NULL DEFAULT 0,
  `n_succeeded` INT NOT NULL DEFAULT 0,
  `n_failed` INT NOT NULL DEFAULT 0,
  `n_cancelled` INT NOT NULL DEFAULT 0,
  PRIMARY KEY (`id`),
  FOREIGN KEY (`id`) REFERENCES batches(id) ON DELETE CASCADE
) ENGINE = InnoDB;

CREATE TABLE IF NOT EXISTS `batches_cancelled` (
  `id` BIGINT NOT NULL,
  `job_group_id` INT NOT NULL DEFAULT 0,
  PRIMARY KEY (`id`),
  FOREIGN KEY (`id`) REFERENCES batches(id) ON DELETE CASCADE
) ENGINE = InnoDB;

CREATE TABLE IF NOT EXISTS `batches_inst_coll_staging` (
  `batch_id` BIGINT NOT NULL,
  `update_id` INT NOT NULL,
  `job_group_id` INT NOT NULL DEFAULT 0,
  `inst_coll` VARCHAR(255),
  `token` INT NOT NULL,
  `n_jobs` INT NOT NULL DEFAULT 0,
  `n_ready_jobs` INT NOT NULL DEFAULT 0,
  `ready_cores_mcpu` BIGINT NOT NULL DEFAULT 0,
  PRIMARY KEY (`batch_id`, `update_id`, `inst_coll`, `token`),
  FOREIGN KEY (`batch_id`) REFERENCES batches(`id`) ON DELETE CASCADE,
  FOREIGN KEY (`batch_id`, `update_id`) REFERENCES batch_updates (`batch_id`, `update_id`) ON DELETE CASCADE,
  FOREIGN KEY (`inst_coll`) REFERENCES inst_colls(name) ON DELETE CASCADE
) ENGINE = InnoDB;
CREATE INDEX `batches_inst_coll_staging_inst_coll` ON `batches_inst_coll_staging` (`inst_coll`);
CREATE INDEX batches_inst_coll_staging_batch_id_jg_id ON batches_inst_coll_staging (`batch_id`, `job_group_id`);

CREATE TABLE `batch_inst_coll_cancellable_resources` (
  `batch_id` BIGINT NOT NULL,
  `update_id` INT NOT NULL,
  `job_group_id` INT NOT NULL DEFAULT 0,
  `inst_coll` VARCHAR(255),
  `token` INT NOT NULL,
  # neither run_always nor cancelled
  `n_ready_cancellable_jobs` INT NOT NULL DEFAULT 0,
  `ready_cancellable_cores_mcpu` BIGINT NOT NULL DEFAULT 0,
  `n_creating_cancellable_jobs` INT NOT NULL DEFAULT 0,
  `n_running_cancellable_jobs` INT NOT NULL DEFAULT 0,
  `running_cancellable_cores_mcpu` BIGINT NOT NULL DEFAULT 0,
  PRIMARY KEY (`batch_id`, `update_id`, `inst_coll`, `token`),
  FOREIGN KEY (`batch_id`) REFERENCES batches(id) ON DELETE CASCADE,
  FOREIGN KEY (`batch_id`, `update_id`) REFERENCES batch_updates (`batch_id`, `update_id`) ON DELETE CASCADE,
  FOREIGN KEY (`inst_coll`) REFERENCES inst_colls(name) ON DELETE CASCADE
) ENGINE = InnoDB;
CREATE INDEX `batch_inst_coll_cancellable_resources_inst_coll` ON `batch_inst_coll_cancellable_resources` (`inst_coll`);
CREATE INDEX batch_inst_coll_cancellable_resources_jg_id ON `batch_inst_coll_cancellable_resources` (`batch_id`, `job_group_id`);

CREATE TABLE IF NOT EXISTS `jobs` (
  `batch_id` BIGINT NOT NULL,
  `job_id` INT NOT NULL,
  `update_id` INT NOT NULL,
  `state` VARCHAR(40) NOT NULL,
  `spec` MEDIUMTEXT NOT NULL,
  `always_run` BOOLEAN NOT NULL,
  `cores_mcpu` INT NOT NULL,
  `status` TEXT,
  `n_pending_parents` INT NOT NULL,
  `cancelled` BOOLEAN NOT NULL DEFAULT FALSE,
  `msec_mcpu` BIGINT NOT NULL DEFAULT 0,
  `attempt_id` VARCHAR(40),
  `inst_coll` VARCHAR(255),
  `n_regions` INT DEFAULT NULL,
  `regions_bits_rep` BIGINT DEFAULT NULL,
  PRIMARY KEY (`batch_id`, `job_id`),
  FOREIGN KEY (`batch_id`) REFERENCES batches(id) ON DELETE CASCADE,
  FOREIGN KEY (`batch_id`, `update_id`) REFERENCES batch_updates(batch_id, update_id) ON DELETE CASCADE,
  FOREIGN KEY (`inst_coll`) REFERENCES inst_colls(name) ON DELETE CASCADE
) ENGINE = InnoDB;
CREATE INDEX `jobs_batch_id_state_always_run_inst_coll_cancelled` ON `jobs` (`batch_id`, `state`, `always_run`, `inst_coll`, `cancelled`);
CREATE INDEX `jobs_batch_id_state_always_run_cancelled` ON `jobs` (`batch_id`, `state`, `always_run`, `cancelled`);
CREATE INDEX `jobs_batch_id_update_id` ON `jobs` (`batch_id`, `update_id`);
CREATE INDEX `jobs_batch_id_always_run_n_regions_regions_bits_rep_job_id` ON `jobs` (`batch_id`, `always_run`, `n_regions`, `regions_bits_rep`, `job_id`);
CREATE INDEX `jobs_batch_id_ic_state_ar_n_regions_bits_rep_job_id` ON `jobs` (`batch_id`, `inst_coll`, `state`, `always_run`, `n_regions`, `regions_bits_rep`, `job_id`);
CREATE INDEX jobs_batch_id_job_group_id ON `jobs` (`batch_id`, `job_group_id`);

CREATE TABLE IF NOT EXISTS `jobs_telemetry` (
  `batch_id` BIGINT NOT NULL,
  `job_id` INT NOT NULL,
  `time_ready` BIGINT DEFAULT NULL,
  PRIMARY KEY (`batch_id`, `job_id`),
  FOREIGN KEY (`batch_id`) REFERENCES batches(id) ON DELETE CASCADE
) ENGINE = InnoDB;

CREATE TABLE IF NOT EXISTS `batch_bunches` (
  `batch_id` BIGINT NOT NULL,
  `start_job_id` INT NOT NULL,
  `token` VARCHAR(100) NOT NULL,
  PRIMARY KEY (`batch_id`, `start_job_id`),
  FOREIGN KEY (`batch_id`, `start_job_id`) REFERENCES jobs(batch_id, job_id) ON DELETE CASCADE
) ENGINE = InnoDB;

CREATE TABLE IF NOT EXISTS `attempts` (
  `batch_id` BIGINT NOT NULL,
  `job_id` INT NOT NULL,
  `attempt_id` VARCHAR(40) NOT NULL,
  `instance_name` VARCHAR(100),
  `start_time` BIGINT,
  `rollup_time` BIGINT,
  `end_time` BIGINT,
  `reason` VARCHAR(40),
  PRIMARY KEY (`batch_id`, `job_id`, `attempt_id`),
  FOREIGN KEY (`batch_id`) REFERENCES batches(id) ON DELETE CASCADE,
  FOREIGN KEY (`batch_id`, `job_id`) REFERENCES jobs(batch_id, job_id) ON DELETE CASCADE,
  FOREIGN KEY (`instance_name`) REFERENCES instances(name) ON DELETE CASCADE
) ENGINE = InnoDB;
CREATE INDEX `attempts_instance_name` ON `attempts` (`instance_name`);
CREATE INDEX `attempts_start_time` ON `attempts` (`start_time`);
CREATE INDEX `attempts_end_time` ON `attempts` (`end_time`);

CREATE TABLE IF NOT EXISTS `gevents_mark` (
  mark VARCHAR(40)
) ENGINE = InnoDB;

INSERT INTO `gevents_mark` (mark) VALUES (NULL);

CREATE TABLE IF NOT EXISTS `job_parents` (
  `batch_id` BIGINT NOT NULL,
  `job_id` INT NOT NULL,
  `parent_id` INT NOT NULL,
  PRIMARY KEY (`batch_id`, `job_id`, `parent_id`),
  FOREIGN KEY (`batch_id`) REFERENCES batches(id) ON DELETE CASCADE,
  FOREIGN KEY (`batch_id`, `job_id`) REFERENCES jobs(batch_id, job_id) ON DELETE CASCADE
) ENGINE = InnoDB;
CREATE INDEX job_parents_parent_id ON `job_parents` (batch_id, parent_id);

CREATE TABLE IF NOT EXISTS `job_attributes` (
  `batch_id` BIGINT NOT NULL,
  `job_id` INT NOT NULL,
  `key` VARCHAR(100) NOT NULL,
  `value` TEXT,
  PRIMARY KEY (`batch_id`, `job_id`, `key`),
  FOREIGN KEY (`batch_id`) REFERENCES batches(id) ON DELETE CASCADE,
  FOREIGN KEY (`batch_id`, `job_id`) REFERENCES jobs(batch_id, job_id) ON DELETE CASCADE
) ENGINE = InnoDB;
CREATE INDEX job_attributes_key_value ON `job_attributes` (`key`, `value`(256));
CREATE INDEX job_attributes_batch_id_key_value ON `job_attributes` (batch_id, `key`, `value`(256));
CREATE INDEX job_attributes_value ON `job_attributes` (batch_id, `value`(256));

CREATE TABLE IF NOT EXISTS `regions` (
  `region_id` INT NOT NULL AUTO_INCREMENT,
  `region` VARCHAR(40) NOT NULL,
  PRIMARY KEY (`region_id`),
  UNIQUE(region)
) ENGINE = InnoDB;

CREATE TABLE IF NOT EXISTS `batch_attributes` (
  `batch_id` BIGINT NOT NULL,
  `job_group_id` INT NOT NULL DEFAULT 0,
  `key` VARCHAR(100) NOT NULL,
  `value` TEXT,
  PRIMARY KEY (`batch_id`, `key`),
  FOREIGN KEY (`batch_id`) REFERENCES batches(id) ON DELETE CASCADE
) ENGINE = InnoDB;
CREATE INDEX batch_attributes_key_value ON `batch_attributes` (`key`, `value`(256));
CREATE INDEX batch_attributes_value ON `batch_attributes` (`value`(256));
CREATE INDEX batch_attributes_batch_id_key_value ON `batch_attributes` (`batch_id`, `job_group_id`, `key`, `value`(256));
CREATE INDEX batch_attributes_batch_id_value ON `batch_attributes` (`batch_id`, `job_group_id`, `value`(256));

DROP TABLE IF EXISTS `aggregated_billing_project_user_resources_v2`;
CREATE TABLE IF NOT EXISTS `aggregated_billing_project_user_resources_v2` (
  `billing_project` VARCHAR(100) NOT NULL,
  `user` VARCHAR(100) NOT NULL,
  `resource_id` INT NOT NULL,
  `token` INT NOT NULL,
  `usage` BIGINT NOT NULL DEFAULT 0,
  `migrated` BOOLEAN DEFAULT FALSE,
  PRIMARY KEY (`billing_project`, `user`, `resource_id`, `token`),
  FOREIGN KEY (`billing_project`) REFERENCES billing_projects(name) ON DELETE CASCADE,
  FOREIGN KEY (`resource_id`) REFERENCES resources(`resource_id`) ON DELETE CASCADE
) ENGINE = InnoDB;
CREATE INDEX aggregated_billing_project_user_resources_v2 ON `aggregated_billing_project_user_resources_v2` (`user`);

DROP TABLE IF EXISTS `aggregated_billing_project_user_resources_by_date_v2`;
CREATE TABLE IF NOT EXISTS `aggregated_billing_project_user_resources_by_date_v2` (
  `billing_date` DATE NOT NULL,
  `billing_project` VARCHAR(100) NOT NULL,
  `user` VARCHAR(100) NOT NULL,
  `resource_id` INT NOT NULL,
  `token` INT NOT NULL,
  `usage` BIGINT NOT NULL DEFAULT 0,
  `migrated` BOOLEAN DEFAULT FALSE,
  PRIMARY KEY (`billing_date`, `billing_project`, `user`, `resource_id`, `token`),
  FOREIGN KEY (`billing_project`) REFERENCES billing_projects(name) ON DELETE CASCADE,
  FOREIGN KEY (`resource_id`) REFERENCES resources(`resource_id`) ON DELETE CASCADE
) ENGINE = InnoDB;
CREATE INDEX aggregated_billing_project_user_resources_by_date_v2_user ON `aggregated_billing_project_user_resources_by_date_v2` (`billing_date`, `user`);

DROP TABLE IF EXISTS `aggregated_batch_resources_v2`;
CREATE TABLE IF NOT EXISTS `aggregated_batch_resources_v2` (
  `batch_id` BIGINT NOT NULL,
  `job_group_id` INT NOT NULL DEFAULT 0,
  `resource_id` INT NOT NULL,
  `token` INT NOT NULL,
  `usage` BIGINT NOT NULL DEFAULT 0,
  `migrated` BOOLEAN DEFAULT FALSE,
  PRIMARY KEY (`batch_id`, `resource_id`, `token`),
  FOREIGN KEY (`batch_id`) REFERENCES batches(`id`) ON DELETE CASCADE,
  FOREIGN KEY (`resource_id`) REFERENCES resources(`resource_id`) ON DELETE CASCADE
) ENGINE = InnoDB;

DROP TABLE IF EXISTS `aggregated_job_resources_v2`;
CREATE TABLE IF NOT EXISTS `aggregated_job_resources_v2` (
  `batch_id` BIGINT NOT NULL,
  `job_id` INT NOT NULL,
  `resource_id` INT NOT NULL,
  `usage` BIGINT NOT NULL DEFAULT 0,
  `migrated` BOOLEAN DEFAULT FALSE,
  PRIMARY KEY (`batch_id`, `job_id`, `resource_id`),
  FOREIGN KEY (`batch_id`) REFERENCES batches(`id`) ON DELETE CASCADE,
  FOREIGN KEY (`batch_id`, `job_id`) REFERENCES jobs(`batch_id`, `job_id`) ON DELETE CASCADE,
  FOREIGN KEY (`resource_id`) REFERENCES resources(`resource_id`) ON DELETE CASCADE
) ENGINE = InnoDB;

CREATE TABLE IF NOT EXISTS `aggregated_billing_project_user_resources_v3` (
  `billing_project` VARCHAR(100) NOT NULL,
  `user` VARCHAR(100) NOT NULL,
  `resource_id` INT NOT NULL,
  `token` INT NOT NULL,
  `usage` BIGINT NOT NULL DEFAULT 0,
  PRIMARY KEY (`billing_project`, `user`, `resource_id`, `token`),
  FOREIGN KEY (`billing_project`) REFERENCES billing_projects(name) ON DELETE CASCADE,
  FOREIGN KEY (`resource_id`) REFERENCES resources(`resource_id`) ON DELETE CASCADE
) ENGINE = InnoDB;
CREATE INDEX aggregated_billing_project_user_resources_v3 ON `aggregated_billing_project_user_resources_v3` (`user`);
CREATE INDEX aggregated_billing_project_user_resources_v3_token ON `aggregated_billing_project_user_resources_v3` (`token`);

CREATE TABLE IF NOT EXISTS `aggregated_billing_project_user_resources_by_date_v3` (
  `billing_date` DATE NOT NULL,
  `billing_project` VARCHAR(100) NOT NULL,
  `user` VARCHAR(100) NOT NULL,
  `resource_id` INT NOT NULL,
  `token` INT NOT NULL,
  `usage` BIGINT NOT NULL DEFAULT 0,
  PRIMARY KEY (`billing_date`, `billing_project`, `user`, `resource_id`, `token`),
  FOREIGN KEY (`billing_project`) REFERENCES billing_projects(name) ON DELETE CASCADE,
  FOREIGN KEY (`resource_id`) REFERENCES resources(`resource_id`) ON DELETE CASCADE
) ENGINE = InnoDB;
CREATE INDEX aggregated_billing_project_user_resources_by_date_v3_user ON `aggregated_billing_project_user_resources_by_date_v3` (`billing_date`, `user`);
CREATE INDEX aggregated_billing_project_user_resources_by_date_v3_token ON `aggregated_billing_project_user_resources_by_date_v3` (`token`);

CREATE TABLE IF NOT EXISTS `aggregated_batch_resources_v3` (
  `batch_id` BIGINT NOT NULL,
  `job_group_id` INT NOT NULL DEFAULT 0,
  `resource_id` INT NOT NULL,
  `token` INT NOT NULL,
  `usage` BIGINT NOT NULL DEFAULT 0,
  PRIMARY KEY (`batch_id`, `resource_id`, `token`),
  FOREIGN KEY (`batch_id`) REFERENCES batches(`id`) ON DELETE CASCADE,
  FOREIGN KEY (`resource_id`) REFERENCES resources(`resource_id`) ON DELETE CASCADE
) ENGINE = InnoDB;

CREATE TABLE IF NOT EXISTS `aggregated_job_resources_v3` (
  `batch_id` BIGINT NOT NULL,
  `job_id` INT NOT NULL,
  `resource_id` INT NOT NULL,
  `usage` BIGINT NOT NULL DEFAULT 0,
  PRIMARY KEY (`batch_id`, `job_id`, `resource_id`),
  FOREIGN KEY (`batch_id`) REFERENCES batches(`id`) ON DELETE CASCADE,
  FOREIGN KEY (`batch_id`, `job_id`) REFERENCES jobs(`batch_id`, `job_id`) ON DELETE CASCADE,
  FOREIGN KEY (`resource_id`) REFERENCES resources(`resource_id`) ON DELETE CASCADE
) ENGINE = InnoDB;

CREATE TABLE IF NOT EXISTS `attempt_resources` (
  `batch_id` BIGINT NOT NULL,
  `job_id` INT NOT NULL,
  `attempt_id` VARCHAR(40) NOT NULL,
  `quantity` BIGINT NOT NULL,
  `resource_id` INT NOT NULL,
  `deduped_resource_id` INT DEFAULT NULL
  PRIMARY KEY (`batch_id`, `job_id`, `attempt_id`, `resource_id`),
  FOREIGN KEY (`batch_id`) REFERENCES batches(`id`) ON DELETE CASCADE,
  FOREIGN KEY (`batch_id`, `job_id`) REFERENCES jobs(`batch_id`, `job_id`) ON DELETE CASCADE,
  FOREIGN KEY (`batch_id`, `job_id`, `attempt_id`) REFERENCES attempts(`batch_id`, `job_id`, `attempt_id`) ON DELETE CASCADE,
  FOREIGN KEY (`resource_id`) REFERENCES resources(`resource_id`) ON DELETE CASCADE
) ENGINE = InnoDB;

DELIMITER $$

DROP TRIGGER IF EXISTS instances_before_update $$
CREATE TRIGGER instances_before_update BEFORE UPDATE on instances
FOR EACH ROW
BEGIN
  IF OLD.time_deactivated IS NOT NULL AND (NEW.time_deactivated IS NULL OR NEW.time_deactivated > OLD.time_deactivated) THEN
    SET NEW.time_deactivated = OLD.time_deactivated;
  END IF;
END $$

DROP TRIGGER IF EXISTS attempts_before_update $$
CREATE TRIGGER attempts_before_update BEFORE UPDATE ON attempts
FOR EACH ROW
BEGIN
  IF OLD.start_time IS NOT NULL AND (NEW.start_time IS NULL OR OLD.start_time < NEW.start_time) THEN
    SET NEW.start_time = OLD.start_time;
  END IF;

  # for job private instances that do not finish creating
  IF NEW.reason = 'activation_timeout' THEN
    SET NEW.start_time = NULL;
  END IF;

  IF OLD.reason IS NOT NULL AND (OLD.end_time IS NULL OR NEW.end_time IS NULL OR NEW.end_time >= OLD.end_time) THEN
    SET NEW.end_time = OLD.end_time;
    SET NEW.reason = OLD.reason;
  END IF;

  # rollup_time should not go backward in time
  # this could happen if MJS happens after the billing update is received
  IF NEW.rollup_time IS NOT NULL AND OLD.rollup_time IS NOT NULL AND NEW.rollup_time < OLD.rollup_time THEN
    SET NEW.rollup_time = OLD.rollup_time;
  END IF;

  # rollup_time should never be less than the start time
  IF NEW.rollup_time IS NOT NULL AND NEW.start_time IS NOT NULL AND NEW.rollup_time < NEW.start_time THEN
    SET NEW.rollup_time = OLD.rollup_time;
  END IF;

  # rollup_time should never be greater than the end time
  IF NEW.rollup_time IS NOT NULL AND NEW.end_time IS NOT NULL AND NEW.rollup_time > NEW.end_time THEN
    SET NEW.rollup_time = NEW.end_time;
  END IF;
END $$

DROP TRIGGER IF EXISTS attempts_after_update $$
CREATE TRIGGER attempts_after_update AFTER UPDATE ON attempts
FOR EACH ROW
BEGIN
  DECLARE job_cores_mcpu INT;
  DECLARE cur_billing_project VARCHAR(100);
  DECLARE msec_diff_rollup BIGINT;
  DECLARE cur_n_tokens INT;
  DECLARE rand_token INT;
  DECLARE cur_billing_date DATE;

  SELECT n_tokens INTO cur_n_tokens FROM globals LOCK IN SHARE MODE;
  SET rand_token = FLOOR(RAND() * cur_n_tokens);

  SELECT cores_mcpu INTO job_cores_mcpu FROM jobs
  WHERE batch_id = NEW.batch_id AND job_id = NEW.job_id;

  SELECT billing_project INTO cur_billing_project FROM batches WHERE id = NEW.batch_id;

  SET msec_diff_rollup = (GREATEST(COALESCE(NEW.rollup_time - NEW.start_time, 0), 0) -
                          GREATEST(COALESCE(OLD.rollup_time - OLD.start_time, 0), 0));

  SET cur_billing_date = CAST(UTC_DATE() AS DATE);

  IF msec_diff_rollup != 0 THEN
    INSERT INTO aggregated_billing_project_user_resources_v2 (billing_project, user, resource_id, token, `usage`)
    SELECT billing_project, `user`,
      resource_id,
      rand_token,
      msec_diff_rollup * quantity
    FROM attempt_resources
    JOIN batches ON batches.id = attempt_resources.batch_id
    WHERE batch_id = NEW.batch_id AND job_id = NEW.job_id AND attempt_id = NEW.attempt_id
    ON DUPLICATE KEY UPDATE `usage` = `usage` + msec_diff_rollup * quantity;

    INSERT INTO aggregated_billing_project_user_resources_v3 (billing_project, user, resource_id, token, `usage`)
    SELECT batches.billing_project, batches.`user`,
      attempt_resources.deduped_resource_id,
      rand_token,
      msec_diff_rollup * quantity
    FROM attempt_resources
    JOIN batches ON batches.id = attempt_resources.batch_id
    INNER JOIN aggregated_billing_project_user_resources_v2 ON
      aggregated_billing_project_user_resources_v2.billing_project = batches.billing_project AND
      aggregated_billing_project_user_resources_v2.user = batches.user AND
      aggregated_billing_project_user_resources_v2.resource_id = attempt_resources.resource_id AND
      aggregated_billing_project_user_resources_v2.token = rand_token
    WHERE attempt_resources.batch_id = NEW.batch_id AND attempt_resources.job_id = NEW.job_id AND attempt_id = NEW.attempt_id AND migrated = 1
    ON DUPLICATE KEY UPDATE `usage` = aggregated_billing_project_user_resources_v3.`usage` + msec_diff_rollup * quantity;

    INSERT INTO aggregated_batch_resources_v2 (batch_id, resource_id, token, `usage`)
    SELECT batch_id,
      resource_id,
      rand_token,
      msec_diff_rollup * quantity
    FROM attempt_resources
    WHERE batch_id = NEW.batch_id AND job_id = NEW.job_id AND attempt_id = NEW.attempt_id
    ON DUPLICATE KEY UPDATE `usage` = `usage` + msec_diff_rollup * quantity;

    INSERT INTO aggregated_batch_resources_v3 (batch_id, resource_id, token, `usage`)
    SELECT attempt_resources.batch_id,
      attempt_resources.deduped_resource_id,
      rand_token,
      msec_diff_rollup * quantity
    FROM attempt_resources
    JOIN aggregated_batch_resources_v2 ON
      aggregated_batch_resources_v2.batch_id = attempt_resources.batch_id AND
      aggregated_batch_resources_v2.resource_id = attempt_resources.resource_id AND
      aggregated_batch_resources_v2.token = rand_token
    WHERE attempt_resources.batch_id = NEW.batch_id AND attempt_resources.job_id = NEW.job_id AND attempt_id = NEW.attempt_id AND migrated = 1
    ON DUPLICATE KEY UPDATE `usage` = aggregated_batch_resources_v3.`usage` + msec_diff_rollup * quantity;

    INSERT INTO aggregated_job_resources_v2 (batch_id, job_id, resource_id, `usage`)
    SELECT batch_id, job_id,
      resource_id,
      msec_diff_rollup * quantity
    FROM attempt_resources
    WHERE batch_id = NEW.batch_id AND job_id = NEW.job_id AND attempt_id = NEW.attempt_id
    ON DUPLICATE KEY UPDATE `usage` = `usage` + msec_diff_rollup * quantity;

    INSERT INTO aggregated_job_resources_v3 (batch_id, job_id, resource_id, `usage`)
    SELECT attempt_resources.batch_id, attempt_resources.job_id,
      attempt_resources.deduped_resource_id,
      msec_diff_rollup * quantity
    FROM attempt_resources
    JOIN aggregated_job_resources_v2 ON
      aggregated_job_resources_v2.batch_id = attempt_resources.batch_id AND
      aggregated_job_resources_v2.job_id = attempt_resources.job_id AND
      aggregated_job_resources_v2.resource_id = attempt_resources.resource_id
    WHERE attempt_resources.batch_id = NEW.batch_id AND attempt_resources.job_id = NEW.job_id AND attempt_id = NEW.attempt_id AND migrated = 1
    ON DUPLICATE KEY UPDATE `usage` = aggregated_job_resources_v3.`usage` + msec_diff_rollup * quantity;

    INSERT INTO aggregated_billing_project_user_resources_by_date_v2 (billing_date, billing_project, user, resource_id, token, `usage`)
    SELECT cur_billing_date,
      billing_project,
      `user`,
      resource_id,
      rand_token,
      msec_diff_rollup * quantity
    FROM attempt_resources
    JOIN batches ON batches.id = attempt_resources.batch_id
    WHERE batch_id = NEW.batch_id AND job_id = NEW.job_id AND attempt_id = NEW.attempt_id
    ON DUPLICATE KEY UPDATE `usage` = `usage` + msec_diff_rollup * quantity;

    INSERT INTO aggregated_billing_project_user_resources_by_date_v3 (billing_date, billing_project, user, resource_id, token, `usage`)
    SELECT cur_billing_date,
      batches.billing_project,
      batches.`user`,
      attempt_resources.deduped_resource_id,
      rand_token,
      msec_diff_rollup * quantity
    FROM attempt_resources
    JOIN batches ON batches.id = attempt_resources.batch_id
    JOIN aggregated_billing_project_user_resources_by_date_v2 ON
      aggregated_billing_project_user_resources_by_date_v2.billing_date = cur_billing_date AND
      aggregated_billing_project_user_resources_by_date_v2.billing_project = batches.billing_project AND
      aggregated_billing_project_user_resources_by_date_v2.user = batches.user AND
      aggregated_billing_project_user_resources_by_date_v2.resource_id = attempt_resources.resource_id AND
      aggregated_billing_project_user_resources_by_date_v2.token = rand_token
    WHERE attempt_resources.batch_id = NEW.batch_id AND attempt_resources.job_id = NEW.job_id AND attempt_id = NEW.attempt_id AND migrated = 1
    ON DUPLICATE KEY UPDATE `usage` = aggregated_billing_project_user_resources_by_date_v3.`usage` + msec_diff_rollup * quantity;
  END IF;
END $$

DROP TRIGGER IF EXISTS batches_after_update $$
CREATE TRIGGER batches_after_update AFTER UPDATE ON batches
FOR EACH ROW
BEGIN
  IF OLD.migrated_batch = 0 AND NEW.migrated_batch = 1 THEN
    INSERT INTO job_groups (batch_id, job_group_id, `user`, cancel_after_n_failures, `state`, n_jobs, time_created, time_completed, callback, attributes)
    VALUES (NEW.id, 0, NEW.`user`, NEW.cancel_after_n_failures, NEW.state, NEW.n_jobs, NEW.time_created, NEW.time_completed, NEW.callback, NEW.attributes);

    INSERT INTO job_group_parents (batch_id, job_group_id, parent_id, `level`)
    VALUES (NEW.id, 0, 0, 0);
  END IF;
END $$

DROP TRIGGER IF EXISTS jobs_after_update $$
CREATE TRIGGER jobs_after_update AFTER UPDATE ON jobs
FOR EACH ROW
BEGIN
  DECLARE cur_user VARCHAR(100);
  DECLARE cur_batch_cancelled BOOLEAN;
  DECLARE cur_n_tokens INT;
  DECLARE rand_token INT;

  DECLARE always_run boolean;
  DECLARE cores_mcpu bigint;

  DECLARE was_marked_cancelled boolean;
  DECLARE was_cancelled        boolean;
  DECLARE was_cancellable      boolean;

  DECLARE now_marked_cancelled boolean;
  DECLARE now_cancelled        boolean;
  DECLARE now_cancellable      boolean;

  DECLARE was_ready boolean;
  DECLARE now_ready boolean;

  DECLARE was_running boolean;
  DECLARE now_running boolean;

  DECLARE was_creating boolean;
  DECLARE now_creating boolean;

  DECLARE delta_n_ready_cancellable_jobs          int;
  DECLARE delta_ready_cancellable_cores_mcpu   bigint;
  DECLARE delta_n_ready_jobs                      int;
  DECLARE delta_ready_cores_mcpu               bigint;
  DECLARE delta_n_cancelled_ready_jobs            int;

  DECLARE delta_n_running_cancellable_jobs        int;
  DECLARE delta_running_cancellable_cores_mcpu bigint;
  DECLARE delta_n_running_jobs                    int;
  DECLARE delta_running_cores_mcpu             bigint;
  DECLARE delta_n_cancelled_running_jobs          int;

  DECLARE delta_n_creating_cancellable_jobs       int;
  DECLARE delta_n_creating_jobs                   int;
  DECLARE delta_n_cancelled_creating_jobs         int;

  SELECT user INTO cur_user FROM batches WHERE id = NEW.batch_id;

  SET cur_batch_cancelled = EXISTS (SELECT TRUE
                                    FROM batches_cancelled
                                    WHERE id = NEW.batch_id
                                    LOCK IN SHARE MODE);

  SELECT n_tokens INTO cur_n_tokens FROM globals LOCK IN SHARE MODE;
  SET rand_token = FLOOR(RAND() * cur_n_tokens);

  SET always_run = old.always_run; # always_run is immutable
  SET cores_mcpu = old.cores_mcpu; # cores_mcpu is immutable

  SET was_marked_cancelled = old.cancelled OR cur_batch_cancelled;
  SET was_cancelled        = NOT always_run AND was_marked_cancelled;
  SET was_cancellable      = NOT always_run AND NOT was_marked_cancelled;

  SET now_marked_cancelled = new.cancelled or cur_batch_cancelled;
  SET now_cancelled        = NOT always_run AND now_marked_cancelled;
  SET now_cancellable      = NOT always_run AND NOT now_marked_cancelled;

  # NB: was_cancelled => now_cancelled b/c you cannot be uncancelled

  SET was_ready    = old.state = 'Ready';
  SET now_ready    = new.state = 'Ready';
  SET was_running  = old.state = 'Running';
  SET now_running  = new.state = 'Running';
  SET was_creating = old.state = 'Creating';
  SET now_creating = new.state = 'Creating';

  SET delta_n_ready_cancellable_jobs        = (-1 * was_ready    *  was_cancellable  )     + (now_ready    *  now_cancellable  ) ;
  SET delta_n_ready_jobs                    = (-1 * was_ready    * (NOT was_cancelled))    + (now_ready    * (NOT now_cancelled));
  SET delta_n_cancelled_ready_jobs          = (-1 * was_ready    *  was_cancelled    )     + (now_ready    *  now_cancelled    ) ;

  SET delta_n_running_cancellable_jobs      = (-1 * was_running  *  was_cancellable  )     + (now_running  *  now_cancellable  ) ;
  SET delta_n_running_jobs                  = (-1 * was_running  * (NOT was_cancelled))    + (now_running  * (NOT now_cancelled));
  SET delta_n_cancelled_running_jobs        = (-1 * was_running  *  was_cancelled    )     + (now_running  *  now_cancelled    ) ;

  SET delta_n_creating_cancellable_jobs     = (-1 * was_creating *  was_cancellable  )     + (now_creating *  now_cancellable  ) ;
  SET delta_n_creating_jobs                 = (-1 * was_creating * (NOT was_cancelled))    + (now_creating * (NOT now_cancelled));
  SET delta_n_cancelled_creating_jobs       = (-1 * was_creating *  was_cancelled    )     + (now_creating *  now_cancelled    ) ;

  SET delta_ready_cancellable_cores_mcpu    = delta_n_ready_cancellable_jobs * cores_mcpu;
  SET delta_ready_cores_mcpu                = delta_n_ready_jobs * cores_mcpu;

  SET delta_running_cancellable_cores_mcpu  = delta_n_running_cancellable_jobs * cores_mcpu;
  SET delta_running_cores_mcpu              = delta_n_running_jobs * cores_mcpu;

  INSERT INTO batch_inst_coll_cancellable_resources (batch_id, update_id, inst_coll, token,
    n_ready_cancellable_jobs,
    ready_cancellable_cores_mcpu,
    n_creating_cancellable_jobs,
    n_running_cancellable_jobs,
    running_cancellable_cores_mcpu)
  VALUES (NEW.batch_id, NEW.update_id, NEW.inst_coll, rand_token,
    delta_n_ready_cancellable_jobs,
    delta_ready_cancellable_cores_mcpu,
    delta_n_creating_cancellable_jobs,
    delta_n_running_cancellable_jobs,
    delta_running_cancellable_cores_mcpu)
  ON DUPLICATE KEY UPDATE
    n_ready_cancellable_jobs = n_ready_cancellable_jobs + delta_n_ready_cancellable_jobs,
    ready_cancellable_cores_mcpu = ready_cancellable_cores_mcpu + delta_ready_cancellable_cores_mcpu,
    n_creating_cancellable_jobs = n_creating_cancellable_jobs + delta_n_creating_cancellable_jobs,
    n_running_cancellable_jobs = n_running_cancellable_jobs + delta_n_running_cancellable_jobs,
    running_cancellable_cores_mcpu = running_cancellable_cores_mcpu + delta_running_cancellable_cores_mcpu;

  INSERT INTO user_inst_coll_resources (user, inst_coll, token,
    n_ready_jobs,
    n_running_jobs,
    n_creating_jobs,
    ready_cores_mcpu,
    running_cores_mcpu,
    n_cancelled_ready_jobs,
    n_cancelled_running_jobs,
    n_cancelled_creating_jobs
  )
  VALUES (cur_user, NEW.inst_coll, rand_token,
    delta_n_ready_jobs,
    delta_n_running_jobs,
    delta_n_creating_jobs,
    delta_ready_cores_mcpu,
    delta_running_cores_mcpu,
    delta_n_cancelled_ready_jobs,
    delta_n_cancelled_running_jobs,
    delta_n_cancelled_creating_jobs
  )
  ON DUPLICATE KEY UPDATE
    n_ready_jobs = n_ready_jobs + delta_n_ready_jobs,
    n_running_jobs = n_running_jobs + delta_n_running_jobs,
    n_creating_jobs = n_creating_jobs + delta_n_creating_jobs,
    ready_cores_mcpu = ready_cores_mcpu + delta_ready_cores_mcpu,
    running_cores_mcpu = running_cores_mcpu + delta_running_cores_mcpu,
    n_cancelled_ready_jobs = n_cancelled_ready_jobs + delta_n_cancelled_ready_jobs,
    n_cancelled_running_jobs = n_cancelled_running_jobs + delta_n_cancelled_running_jobs,
    n_cancelled_creating_jobs = n_cancelled_creating_jobs + delta_n_cancelled_creating_jobs;
END $$

DROP TRIGGER IF EXISTS attempt_resources_after_insert $$
CREATE TRIGGER attempt_resources_after_insert AFTER INSERT ON attempt_resources
FOR EACH ROW
BEGIN
  DECLARE cur_start_time BIGINT;
  DECLARE cur_rollup_time BIGINT;
  DECLARE cur_billing_project VARCHAR(100);
  DECLARE cur_user VARCHAR(100);
  DECLARE msec_diff_rollup BIGINT;
  DECLARE cur_n_tokens INT;
  DECLARE rand_token INT;
  DECLARE cur_billing_date DATE;
  DECLARE bp_user_resources_migrated BOOLEAN DEFAULT FALSE;
  DECLARE bp_user_resources_by_date_migrated BOOLEAN DEFAULT FALSE;
  DECLARE batch_resources_migrated BOOLEAN DEFAULT FALSE;
  DECLARE job_resources_migrated BOOLEAN DEFAULT FALSE;

  SELECT billing_project, user INTO cur_billing_project, cur_user
  FROM batches WHERE id = NEW.batch_id;

  SELECT n_tokens INTO cur_n_tokens FROM globals LOCK IN SHARE MODE;
  SET rand_token = FLOOR(RAND() * cur_n_tokens);

  SELECT start_time, rollup_time INTO cur_start_time, cur_rollup_time
  FROM attempts
  WHERE batch_id = NEW.batch_id AND job_id = NEW.job_id AND attempt_id = NEW.attempt_id
  LOCK IN SHARE MODE;

  SET msec_diff_rollup = GREATEST(COALESCE(cur_rollup_time - cur_start_time, 0), 0);

  SET cur_billing_date = CAST(UTC_DATE() AS DATE);

  IF msec_diff_rollup != 0 THEN
    INSERT INTO aggregated_billing_project_user_resources_v2 (billing_project, user, resource_id, token, `usage`)
    VALUES (cur_billing_project, cur_user, NEW.resource_id, rand_token, NEW.quantity * msec_diff_rollup)
    ON DUPLICATE KEY UPDATE
      `usage` = `usage` + NEW.quantity * msec_diff_rollup;

    SELECT migrated INTO bp_user_resources_migrated
    FROM aggregated_billing_project_user_resources_v2
    WHERE billing_project = cur_billing_project AND user = cur_user AND resource_id = NEW.resource_id AND token = rand_token
    FOR UPDATE;

    IF bp_user_resources_migrated THEN
      INSERT INTO aggregated_billing_project_user_resources_v3 (billing_project, user, resource_id, token, `usage`)
      VALUES (cur_billing_project, cur_user, NEW.deduped_resource_id, rand_token, NEW.quantity * msec_diff_rollup)
      ON DUPLICATE KEY UPDATE
        `usage` = `usage` + NEW.quantity * msec_diff_rollup;
    END IF;

    INSERT INTO aggregated_batch_resources_v2 (batch_id, resource_id, token, `usage`)
    VALUES (NEW.batch_id, NEW.resource_id, rand_token, NEW.quantity * msec_diff_rollup)
    ON DUPLICATE KEY UPDATE
      `usage` = `usage` + NEW.quantity * msec_diff_rollup;

    SELECT migrated INTO batch_resources_migrated
    FROM aggregated_batch_resources_v2
    WHERE batch_id = NEW.batch_id AND resource_id = NEW.resource_id AND token = rand_token
    FOR UPDATE;

    IF batch_resources_migrated THEN
      INSERT INTO aggregated_batch_resources_v3 (batch_id, resource_id, token, `usage`)
      VALUES (NEW.batch_id, NEW.deduped_resource_id, rand_token, NEW.quantity * msec_diff_rollup)
      ON DUPLICATE KEY UPDATE
        `usage` = `usage` + NEW.quantity * msec_diff_rollup;
    END IF;

    INSERT INTO aggregated_job_resources_v2 (batch_id, job_id, resource_id, `usage`)
    VALUES (NEW.batch_id, NEW.job_id, NEW.resource_id, NEW.quantity * msec_diff_rollup)
    ON DUPLICATE KEY UPDATE
      `usage` = `usage` + NEW.quantity * msec_diff_rollup;

    SELECT migrated INTO job_resources_migrated
    FROM aggregated_job_resources_v2
    WHERE batch_id = NEW.batch_id AND job_id = NEW.job_id AND resource_id = NEW.resource_id
    FOR UPDATE;

    IF job_resources_migrated THEN
      INSERT INTO aggregated_job_resources_v3 (batch_id, job_id, resource_id, `usage`)
      VALUES (NEW.batch_id, NEW.job_id, NEW.deduped_resource_id, NEW.quantity * msec_diff_rollup)
      ON DUPLICATE KEY UPDATE
        `usage` = `usage` + NEW.quantity * msec_diff_rollup;
    END IF;

    INSERT INTO aggregated_billing_project_user_resources_by_date_v2 (billing_date, billing_project, user, resource_id, token, `usage`)
    VALUES (cur_billing_date, cur_billing_project, cur_user, NEW.resource_id, rand_token, NEW.quantity * msec_diff_rollup)
    ON DUPLICATE KEY UPDATE
      `usage` = `usage` + NEW.quantity * msec_diff_rollup;

    SELECT migrated INTO bp_user_resources_by_date_migrated
    FROM aggregated_billing_project_user_resources_by_date_v2
    WHERE billing_date = cur_billing_date AND billing_project = cur_billing_project AND user = cur_user
      AND resource_id = NEW.resource_id AND token = rand_token
    FOR UPDATE;

    IF bp_user_resources_by_date_migrated THEN
      INSERT INTO aggregated_billing_project_user_resources_by_date_v3 (billing_date, billing_project, user, resource_id, token, `usage`)
      VALUES (cur_billing_date, cur_billing_project, cur_user, NEW.deduped_resource_id, rand_token, NEW.quantity * msec_diff_rollup)
      ON DUPLICATE KEY UPDATE
        `usage` = `usage` + NEW.quantity * msec_diff_rollup;
    END IF;
  END IF;
END $$

DROP TRIGGER IF EXISTS aggregated_bp_user_resources_v2_before_insert $$
CREATE TRIGGER aggregated_bp_user_resources_v2_before_insert BEFORE INSERT ON aggregated_billing_project_user_resources_v2
FOR EACH ROW
BEGIN
  SET NEW.migrated = 1;
END $$

DROP TRIGGER IF EXISTS aggregated_bp_user_resources_by_date_v2_before_insert $$
CREATE TRIGGER aggregated_bp_user_resources_by_date_v2_before_insert BEFORE INSERT ON aggregated_billing_project_user_resources_by_date_v2
FOR EACH ROW
BEGIN
  SET NEW.migrated = 1;
END $$

DROP TRIGGER IF EXISTS aggregated_batch_resources_v2_before_insert $$
CREATE TRIGGER aggregated_batch_resources_v2_before_insert BEFORE INSERT on aggregated_batch_resources_v2
FOR EACH ROW
BEGIN
  SET NEW.migrated = 1;
END $$

DROP TRIGGER IF EXISTS aggregated_job_resources_v2_before_insert $$
CREATE TRIGGER aggregated_job_resources_v2_before_insert BEFORE INSERT on aggregated_job_resources_v2
FOR EACH ROW
BEGIN
  SET NEW.migrated = 1;
END $$

DROP TRIGGER IF EXISTS aggregated_bp_user_resources_v2_after_update $$
CREATE TRIGGER aggregated_bp_user_resources_v2_after_update AFTER UPDATE ON aggregated_billing_project_user_resources_v2
FOR EACH ROW
BEGIN
  DECLARE new_deduped_resource_id INT;

  IF OLD.migrated = 0 AND NEW.migrated = 1 THEN
    SELECT deduped_resource_id INTO new_deduped_resource_id FROM resources WHERE resource_id = OLD.resource_id;

    INSERT INTO aggregated_billing_project_user_resources_v3 (billing_project, user, resource_id, token, `usage`)
    VALUES (NEW.billing_project, NEW.user, new_deduped_resource_id, NEW.token, NEW.usage)
    ON DUPLICATE KEY UPDATE
      `usage` = `usage` + NEW.usage;
  END IF;
END $$

DROP TRIGGER IF EXISTS aggregated_bp_user_resources_by_date_v2_after_update $$
CREATE TRIGGER aggregated_bp_user_resources_by_date_v2_after_update AFTER UPDATE ON aggregated_billing_project_user_resources_by_date_v2
FOR EACH ROW
BEGIN
  DECLARE new_deduped_resource_id INT;

  IF OLD.migrated = 0 AND NEW.migrated = 1 THEN
    SELECT deduped_resource_id INTO new_deduped_resource_id FROM resources WHERE resource_id = OLD.resource_id;

    INSERT INTO aggregated_billing_project_user_resources_by_date_v3 (billing_date, billing_project, user, resource_id, token, `usage`)
    VALUES (NEW.billing_date, NEW.billing_project, NEW.user, new_deduped_resource_id, NEW.token, NEW.usage)
    ON DUPLICATE KEY UPDATE
        `usage` = `usage` + NEW.usage;
  END IF;
END $$

DROP TRIGGER IF EXISTS aggregated_batch_resources_v2_after_update $$
CREATE TRIGGER aggregated_batch_resources_v2_after_update AFTER UPDATE ON aggregated_batch_resources_v2
FOR EACH ROW
BEGIN
  DECLARE new_deduped_resource_id INT;

  IF OLD.migrated = 0 AND NEW.migrated = 1 THEN
    SELECT deduped_resource_id INTO new_deduped_resource_id FROM resources WHERE resource_id = OLD.resource_id;

    INSERT INTO aggregated_batch_resources_v3 (batch_id, resource_id, token, `usage`)
    VALUES (NEW.batch_id, new_deduped_resource_id, NEW.token, NEW.usage)
    ON DUPLICATE KEY UPDATE
      `usage` = `usage` + NEW.usage;
  END IF;
END $$

DROP TRIGGER IF EXISTS aggregated_job_resources_v2_after_update $$
CREATE TRIGGER aggregated_job_resources_v2_after_update AFTER UPDATE ON aggregated_job_resources_v2
FOR EACH ROW
BEGIN
  DECLARE new_deduped_resource_id INT;

  IF OLD.migrated = 0 AND NEW.migrated = 1 THEN
    SELECT deduped_resource_id INTO new_deduped_resource_id FROM resources WHERE resource_id = OLD.resource_id;

    INSERT INTO aggregated_job_resources_v3 (batch_id, job_id, resource_id, `usage`)
    VALUES (NEW.batch_id, NEW.job_id, new_deduped_resource_id, NEW.usage)
    ON DUPLICATE KEY UPDATE
      `usage` = `usage` + NEW.usage;
  END IF;
END $$

CREATE PROCEDURE activate_instance(
  IN in_instance_name VARCHAR(100),
  IN in_ip_address VARCHAR(100),
  IN in_activation_time BIGINT
)
BEGIN
  DECLARE cur_state VARCHAR(40);
  DECLARE cur_token VARCHAR(100);

  START TRANSACTION;

  SELECT state, token INTO cur_state, cur_token FROM instances
  WHERE name = in_instance_name
  FOR UPDATE;

  IF cur_state = 'pending' THEN
    UPDATE instances
    SET state = 'active',
      activation_token = NULL,
      ip_address = in_ip_address,
      time_activated = in_activation_time WHERE name = in_instance_name;
    COMMIT;
    SELECT 0 as rc, cur_token as token;
  ELSE
    ROLLBACK;
    SELECT 1 as rc, cur_state, 'state not pending' as message;
  END IF;
END $$

DROP PROCEDURE IF EXISTS deactivate_instance $$
CREATE PROCEDURE deactivate_instance(
  IN in_instance_name VARCHAR(100),
  IN in_reason VARCHAR(40),
  IN in_timestamp BIGINT
)
BEGIN
  DECLARE cur_state VARCHAR(40);

  START TRANSACTION;

  SELECT state INTO cur_state FROM instances WHERE name = in_instance_name FOR UPDATE;

  UPDATE instances
  SET time_deactivated = in_timestamp
  WHERE name = in_instance_name;

  UPDATE attempts
  SET rollup_time = in_timestamp, end_time = in_timestamp, reason = in_reason
  WHERE instance_name = in_instance_name;

  IF cur_state = 'pending' or cur_state = 'active' THEN
    UPDATE jobs
    INNER JOIN attempts ON jobs.batch_id = attempts.batch_id AND jobs.job_id = attempts.job_id AND jobs.attempt_id = attempts.attempt_id
    SET state = 'Ready',
        jobs.attempt_id = NULL
    WHERE instance_name = in_instance_name AND (state = 'Running' OR state = 'Creating');

    UPDATE instances, instances_free_cores_mcpu
    SET state = 'inactive',
        free_cores_mcpu = cores_mcpu
    WHERE instances.name = in_instance_name
      AND instances.name = instances_free_cores_mcpu.name;

    COMMIT;
    SELECT 0 as rc;
  ELSE
    ROLLBACK;
    SELECT 1 as rc, cur_state, 'state not live (active or pending)' as message;
  END IF;
END $$

CREATE PROCEDURE mark_instance_deleted(
  IN in_instance_name VARCHAR(100)
)
BEGIN
  DECLARE cur_state VARCHAR(40);

  START TRANSACTION;

  SELECT state INTO cur_state FROM instances WHERE name = in_instance_name FOR UPDATE;

  IF cur_state = 'inactive' THEN
    UPDATE instances SET state = 'deleted' WHERE name = in_instance_name;
    COMMIT;
    SELECT 0 as rc;
  ELSE
    ROLLBACK;
    SELECT 1 as rc, cur_state, 'state not inactive' as message;
  END IF;
END $$

DROP PROCEDURE IF EXISTS commit_batch_update $$
CREATE PROCEDURE commit_batch_update(
  IN in_batch_id BIGINT,
  IN in_update_id INT,
  IN in_timestamp BIGINT
)
BEGIN
  DECLARE cur_update_committed BOOLEAN;
  DECLARE expected_n_jobs INT;
  DECLARE staging_n_jobs INT;
  DECLARE cur_update_start_job_id INT;

  START TRANSACTION;

  SELECT committed, n_jobs INTO cur_update_committed, expected_n_jobs
  FROM batch_updates
  WHERE batch_id = in_batch_id AND update_id = in_update_id
  FOR UPDATE;

  IF cur_update_committed THEN
    COMMIT;
    SELECT 0 as rc;
  ELSE
    SELECT COALESCE(SUM(n_jobs), 0) INTO staging_n_jobs
    FROM batches_inst_coll_staging
    WHERE batch_id = in_batch_id AND update_id = in_update_id AND job_group_id = 0
    FOR UPDATE;

    # we can only check staged equals expected for the root job group
    IF staging_n_jobs = expected_n_jobs THEN
      UPDATE batch_updates
      SET committed = 1, time_committed = in_timestamp
      WHERE batch_id = in_batch_id AND update_id = in_update_id;

      UPDATE batches SET
        `state` = 'running',
        time_completed = NULL,
        n_jobs = n_jobs + expected_n_jobs
      WHERE id = in_batch_id;

      UPDATE job_groups
      INNER JOIN (
        SELECT batch_id, job_group_id, CAST(COALESCE(SUM(n_jobs), 0) AS SIGNED) AS staged_n_jobs
        FROM batches_inst_coll_staging
        WHERE batch_id = in_batch_id AND update_id = in_update_id
        GROUP BY batch_id, job_group_id
      ) AS t ON job_groups.batch_id = t.batch_id AND job_groups.job_group_id = t.job_group_id
      SET `state` = 'running', time_completed = NULL, n_jobs = n_jobs + t.staged_n_jobs;

      # compute global number of new ready jobs from root job group
      INSERT INTO user_inst_coll_resources (user, inst_coll, token, n_ready_jobs, ready_cores_mcpu)
      SELECT user, inst_coll, 0, @n_ready_jobs := COALESCE(SUM(n_ready_jobs), 0), @ready_cores_mcpu := COALESCE(SUM(ready_cores_mcpu), 0)
      FROM batches_inst_coll_staging
      JOIN batches ON batches.id = batches_inst_coll_staging.batch_id
      WHERE batch_id = in_batch_id AND update_id = in_update_id AND job_group_id = 0
      GROUP BY `user`, inst_coll
      ON DUPLICATE KEY UPDATE
        n_ready_jobs = n_ready_jobs + @n_ready_jobs,
        ready_cores_mcpu = ready_cores_mcpu + @ready_cores_mcpu;

      DELETE FROM batches_inst_coll_staging WHERE batch_id = in_batch_id AND update_id = in_update_id;

      IF in_update_id != 1 THEN
        SELECT start_job_id INTO cur_update_start_job_id FROM batch_updates WHERE batch_id = in_batch_id AND update_id = in_update_id;

        UPDATE jobs
          LEFT JOIN `jobs_telemetry` ON `jobs_telemetry`.batch_id = jobs.batch_id AND `jobs_telemetry`.job_id = jobs.job_id
          LEFT JOIN (
            SELECT `job_parents`.batch_id, `job_parents`.job_id,
              COALESCE(SUM(1), 0) AS n_parents,
              COALESCE(SUM(state IN ('Pending', 'Ready', 'Creating', 'Running')), 0) AS n_pending_parents,
              COALESCE(SUM(state = 'Success'), 0) AS n_succeeded
            FROM `job_parents`
            LEFT JOIN `jobs` ON jobs.batch_id = `job_parents`.batch_id AND jobs.job_id = `job_parents`.parent_id
            WHERE job_parents.batch_id = in_batch_id AND
              `job_parents`.job_id >= cur_update_start_job_id AND
              `job_parents`.job_id < cur_update_start_job_id + staging_n_jobs
            GROUP BY `job_parents`.batch_id, `job_parents`.job_id
            FOR UPDATE
          ) AS t
            ON jobs.batch_id = t.batch_id AND
               jobs.job_id = t.job_id
          SET jobs.state = IF(COALESCE(t.n_pending_parents, 0) = 0, 'Ready', 'Pending'),
              jobs.n_pending_parents = COALESCE(t.n_pending_parents, 0),
              jobs.cancelled = IF(COALESCE(t.n_succeeded, 0) = COALESCE(t.n_parents - t.n_pending_parents, 0), jobs.cancelled, 1),
              jobs_telemetry.time_ready = IF(COALESCE(t.n_pending_parents, 0) = 0 AND jobs_telemetry.time_ready IS NULL, in_timestamp, jobs_telemetry.time_ready)
          WHERE jobs.batch_id = in_batch_id AND jobs.job_id >= cur_update_start_job_id AND
              jobs.job_id < cur_update_start_job_id + staging_n_jobs;
      END IF;

      COMMIT;
      SELECT 0 as rc;
    ELSE
      ROLLBACK;
      SELECT 1 as rc, expected_n_jobs, staging_n_jobs as actual_n_jobs, 'wrong number of jobs' as message;
    END IF;
  END IF;
END $$

DROP PROCEDURE IF EXISTS cancel_batch $$
CREATE PROCEDURE cancel_batch(
  IN in_batch_id VARCHAR(100)
)
BEGIN
  DECLARE cur_user VARCHAR(100);
  DECLARE cur_batch_state VARCHAR(40);
  DECLARE cur_cancelled BOOLEAN;
  DECLARE cur_n_cancelled_ready_jobs INT;
  DECLARE cur_cancelled_ready_cores_mcpu BIGINT;
  DECLARE cur_n_cancelled_running_jobs INT;
  DECLARE cur_cancelled_running_cores_mcpu BIGINT;
  DECLARE cur_n_n_cancelled_creating_jobs INT;

  START TRANSACTION;

  SELECT user, `state` INTO cur_user, cur_batch_state FROM batches
  WHERE id = in_batch_id
  FOR UPDATE;

  SET cur_cancelled = EXISTS (SELECT TRUE
                              FROM batches_cancelled
                              WHERE id = in_batch_id
                              FOR UPDATE);

  IF cur_batch_state = 'running' AND NOT cur_cancelled THEN
    INSERT INTO user_inst_coll_resources (user, inst_coll, token,
      n_ready_jobs, ready_cores_mcpu,
      n_running_jobs, running_cores_mcpu,
      n_creating_jobs,
      n_cancelled_ready_jobs, n_cancelled_running_jobs, n_cancelled_creating_jobs)
    SELECT user, inst_coll, 0,
      -1 * (@n_ready_cancellable_jobs := COALESCE(SUM(n_ready_cancellable_jobs), 0)),
      -1 * (@ready_cancellable_cores_mcpu := COALESCE(SUM(ready_cancellable_cores_mcpu), 0)),
      -1 * (@n_running_cancellable_jobs := COALESCE(SUM(n_running_cancellable_jobs), 0)),
      -1 * (@running_cancellable_cores_mcpu := COALESCE(SUM(running_cancellable_cores_mcpu), 0)),
      -1 * (@n_creating_cancellable_jobs := COALESCE(SUM(n_creating_cancellable_jobs), 0)),
      COALESCE(SUM(n_ready_cancellable_jobs), 0),
      COALESCE(SUM(n_running_cancellable_jobs), 0),
      COALESCE(SUM(n_creating_cancellable_jobs), 0)
    FROM batch_inst_coll_cancellable_resources
    JOIN batches ON batches.id = batch_inst_coll_cancellable_resources.batch_id
    INNER JOIN batch_updates ON batch_inst_coll_cancellable_resources.batch_id = batch_updates.batch_id AND
      batch_inst_coll_cancellable_resources.update_id = batch_updates.update_id
    WHERE batch_inst_coll_cancellable_resources.batch_id = in_batch_id AND batch_updates.committed
    GROUP BY user, inst_coll
    ON DUPLICATE KEY UPDATE
      n_ready_jobs = n_ready_jobs - @n_ready_cancellable_jobs,
      ready_cores_mcpu = ready_cores_mcpu - @ready_cancellable_cores_mcpu,
      n_running_jobs = n_running_jobs - @n_running_cancellable_jobs,
      running_cores_mcpu = running_cores_mcpu - @running_cancellable_cores_mcpu,
      n_creating_jobs = n_creating_jobs - @n_creating_cancellable_jobs,
      n_cancelled_ready_jobs = n_cancelled_ready_jobs + @n_ready_cancellable_jobs,
      n_cancelled_running_jobs = n_cancelled_running_jobs + @n_running_cancellable_jobs,
      n_cancelled_creating_jobs = n_cancelled_creating_jobs + @n_creating_cancellable_jobs;

    # there are no cancellable jobs left, they have been cancelled
    DELETE FROM batch_inst_coll_cancellable_resources WHERE batch_id = in_batch_id;

    # cancel root job group only
    INSERT INTO batches_cancelled (id, job_group_id) VALUES (in_batch_id, 0);
  END IF;

  COMMIT;
END $$

DROP PROCEDURE IF EXISTS add_attempt $$
CREATE PROCEDURE add_attempt(
  IN in_batch_id BIGINT,
  IN in_job_id INT,
  IN in_attempt_id VARCHAR(40),
  IN in_instance_name VARCHAR(100),
  IN in_cores_mcpu INT,
  OUT delta_cores_mcpu INT
)
BEGIN
  DECLARE cur_instance_state VARCHAR(40);
  DECLARE dummy_lock INT;

  SET delta_cores_mcpu = IFNULL(delta_cores_mcpu, 0);

  IF in_attempt_id IS NOT NULL THEN
    SELECT 1 INTO dummy_lock FROM instances_free_cores_mcpu
    WHERE instances_free_cores_mcpu.name = in_instance_name
    FOR UPDATE;

    INSERT INTO attempts (batch_id, job_id, attempt_id, instance_name)
    VALUES (in_batch_id, in_job_id, in_attempt_id, in_instance_name)
    ON DUPLICATE KEY UPDATE batch_id = batch_id;

    IF ROW_COUNT() = 1 THEN
      SELECT state INTO cur_instance_state
      FROM instances
      WHERE name = in_instance_name
      LOCK IN SHARE MODE;

      IF cur_instance_state = 'pending' OR cur_instance_state = 'active' THEN
        UPDATE instances_free_cores_mcpu
        SET free_cores_mcpu = free_cores_mcpu - in_cores_mcpu
        WHERE instances_free_cores_mcpu.name = in_instance_name;
      END IF;

      SET delta_cores_mcpu = -1 * in_cores_mcpu;
    END IF;
  END IF;
END $$

DROP PROCEDURE IF EXISTS schedule_job $$
CREATE PROCEDURE schedule_job(
  IN in_batch_id BIGINT,
  IN in_job_id INT,
  IN in_attempt_id VARCHAR(40),
  IN in_instance_name VARCHAR(100)
)
BEGIN
  DECLARE cur_job_state VARCHAR(40);
  DECLARE cur_cores_mcpu INT;
  DECLARE cur_job_cancel BOOLEAN;
  DECLARE cur_instance_state VARCHAR(40);
  DECLARE cur_attempt_id VARCHAR(40);
  DECLARE delta_cores_mcpu INT;
  DECLARE cur_instance_is_pool BOOLEAN;

  START TRANSACTION;

  SELECT state, cores_mcpu, attempt_id
  INTO cur_job_state, cur_cores_mcpu, cur_attempt_id
  FROM jobs
  WHERE batch_id = in_batch_id AND job_id = in_job_id
  FOR UPDATE;

  SELECT (jobs.cancelled OR batches_cancelled.id IS NOT NULL) AND NOT jobs.always_run
  INTO cur_job_cancel
  FROM jobs
  LEFT JOIN batches_cancelled ON batches_cancelled.id = jobs.batch_id
  WHERE batch_id = in_batch_id AND job_id = in_job_id
  LOCK IN SHARE MODE;

  SELECT is_pool
  INTO cur_instance_is_pool
  FROM instances
  LEFT JOIN inst_colls ON instances.inst_coll = inst_colls.name
  WHERE instances.name = in_instance_name;

  CALL add_attempt(in_batch_id, in_job_id, in_attempt_id, in_instance_name, cur_cores_mcpu, delta_cores_mcpu);

  IF cur_instance_is_pool THEN
    IF delta_cores_mcpu = 0 THEN
      SET delta_cores_mcpu = cur_cores_mcpu;
    ELSE
      SET delta_cores_mcpu = 0;
    END IF;
  END IF;

  SELECT state INTO cur_instance_state FROM instances WHERE name = in_instance_name LOCK IN SHARE MODE;

  IF (cur_job_state = 'Ready' OR cur_job_state = 'Creating') AND NOT cur_job_cancel AND cur_instance_state = 'active' THEN
    UPDATE jobs SET state = 'Running', attempt_id = in_attempt_id WHERE batch_id = in_batch_id AND job_id = in_job_id;
    COMMIT;
    SELECT 0 as rc, in_instance_name, delta_cores_mcpu;
  ELSE
    COMMIT;
    SELECT 1 as rc,
      cur_job_state,
      cur_job_cancel,
      cur_instance_state,
      in_instance_name,
      cur_attempt_id,
      delta_cores_mcpu,
      'job not Ready or cancelled or instance not active, but attempt already exists' as message;
  END IF;
END $$

DROP PROCEDURE IF EXISTS unschedule_job $$
CREATE PROCEDURE unschedule_job(
  IN in_batch_id BIGINT,
  IN in_job_id INT,
  IN in_attempt_id VARCHAR(40),
  IN in_instance_name VARCHAR(100),
  IN new_end_time BIGINT,
  IN new_reason VARCHAR(40)
)
BEGIN
  DECLARE cur_job_state VARCHAR(40);
  DECLARE cur_instance_state VARCHAR(40);
  DECLARE cur_attempt_id VARCHAR(40);
  DECLARE cur_cores_mcpu INT;
  DECLARE cur_end_time BIGINT;
  DECLARE delta_cores_mcpu INT DEFAULT 0;

  START TRANSACTION;

  SELECT state, cores_mcpu, attempt_id
  INTO cur_job_state, cur_cores_mcpu, cur_attempt_id
  FROM jobs
  WHERE batch_id = in_batch_id AND job_id = in_job_id
  FOR UPDATE;

  SELECT end_time INTO cur_end_time
  FROM attempts
  WHERE batch_id = in_batch_id AND job_id = in_job_id AND attempt_id = in_attempt_id
  FOR UPDATE;

  UPDATE attempts
  SET rollup_time = new_end_time, end_time = new_end_time, reason = new_reason
  WHERE batch_id = in_batch_id AND job_id = in_job_id AND attempt_id = in_attempt_id;

  SELECT state INTO cur_instance_state FROM instances WHERE name = in_instance_name LOCK IN SHARE MODE;

  IF cur_instance_state = 'active' AND cur_end_time IS NULL THEN
    UPDATE instances_free_cores_mcpu
    SET free_cores_mcpu = free_cores_mcpu + cur_cores_mcpu
    WHERE instances_free_cores_mcpu.name = in_instance_name;

    SET delta_cores_mcpu = cur_cores_mcpu;
  END IF;

  IF (cur_job_state = 'Creating' OR cur_job_state = 'Running') AND cur_attempt_id = in_attempt_id THEN
    UPDATE jobs SET state = 'Ready', attempt_id = NULL WHERE batch_id = in_batch_id AND job_id = in_job_id;
    COMMIT;
    SELECT 0 as rc, delta_cores_mcpu;
  ELSE
    COMMIT;
    SELECT 1 as rc, cur_job_state, delta_cores_mcpu,
      'job state not Running or Creating or wrong attempt id' as message;
  END IF;
END $$

DROP PROCEDURE IF EXISTS mark_job_creating $$
CREATE PROCEDURE mark_job_creating(
  IN in_batch_id BIGINT,
  IN in_job_id INT,
  IN in_attempt_id VARCHAR(40),
  IN in_instance_name VARCHAR(100),
  IN new_start_time BIGINT
)
BEGIN
  DECLARE cur_job_state VARCHAR(40);
  DECLARE cur_job_cancel BOOLEAN;
  DECLARE cur_cores_mcpu INT;
  DECLARE cur_instance_state VARCHAR(40);
  DECLARE delta_cores_mcpu INT;

  START TRANSACTION;

  SELECT state, cores_mcpu
  INTO cur_job_state, cur_cores_mcpu
  FROM jobs
  WHERE batch_id = in_batch_id AND job_id = in_job_id
  FOR UPDATE;

  SELECT (jobs.cancelled OR batches_cancelled.id IS NOT NULL) AND NOT jobs.always_run
  INTO cur_job_cancel
  FROM jobs
  LEFT JOIN batches_cancelled ON batches_cancelled.id = jobs.batch_id
  WHERE batch_id = in_batch_id AND job_id = in_job_id
  LOCK IN SHARE MODE;

  CALL add_attempt(in_batch_id, in_job_id, in_attempt_id, in_instance_name, cur_cores_mcpu, delta_cores_mcpu);

  UPDATE attempts SET start_time = new_start_time, rollup_time = new_start_time
  WHERE batch_id = in_batch_id AND job_id = in_job_id AND attempt_id = in_attempt_id;

  SELECT state INTO cur_instance_state FROM instances WHERE name = in_instance_name LOCK IN SHARE MODE;

  IF cur_job_state = 'Ready' AND NOT cur_job_cancel AND cur_instance_state = 'pending' THEN
    UPDATE jobs SET state = 'Creating', attempt_id = in_attempt_id WHERE batch_id = in_batch_id AND job_id = in_job_id;
  END IF;

  COMMIT;
  SELECT 0 as rc, delta_cores_mcpu;
END $$

DROP PROCEDURE IF EXISTS mark_job_started $$
CREATE PROCEDURE mark_job_started(
  IN in_batch_id BIGINT,
  IN in_job_id INT,
  IN in_attempt_id VARCHAR(40),
  IN in_instance_name VARCHAR(100),
  IN new_start_time BIGINT
)
BEGIN
  DECLARE cur_job_state VARCHAR(40);
  DECLARE cur_job_cancel BOOLEAN;
  DECLARE cur_cores_mcpu INT;
  DECLARE cur_instance_state VARCHAR(40);
  DECLARE delta_cores_mcpu INT;

  START TRANSACTION;

  SELECT state, cores_mcpu
  INTO cur_job_state, cur_cores_mcpu
  FROM jobs
  WHERE batch_id = in_batch_id AND job_id = in_job_id
  FOR UPDATE;

  SELECT (jobs.cancelled OR batches_cancelled.id IS NOT NULL) AND NOT jobs.always_run
  INTO cur_job_cancel
  FROM jobs
  LEFT JOIN batches_cancelled ON batches_cancelled.id = jobs.batch_id
  WHERE batch_id = in_batch_id AND job_id = in_job_id
  LOCK IN SHARE MODE;

  CALL add_attempt(in_batch_id, in_job_id, in_attempt_id, in_instance_name, cur_cores_mcpu, delta_cores_mcpu);

  UPDATE attempts SET start_time = new_start_time, rollup_time = new_start_time
  WHERE batch_id = in_batch_id AND job_id = in_job_id AND attempt_id = in_attempt_id;

  SELECT state INTO cur_instance_state FROM instances WHERE name = in_instance_name LOCK IN SHARE MODE;

  IF cur_job_state = 'Ready' AND NOT cur_job_cancel AND cur_instance_state = 'active' THEN
    UPDATE jobs SET state = 'Running', attempt_id = in_attempt_id WHERE batch_id = in_batch_id AND job_id = in_job_id;
  END IF;

  COMMIT;
  SELECT 0 as rc, delta_cores_mcpu;
END $$

DROP PROCEDURE IF EXISTS mark_job_group_complete $$
CREATE PROCEDURE update_job_groups(
  IN in_batch_id BIGINT,
  IN in_job_group_id INT,
  IN new_timestamp BIGINT
)
BEGIN
  DECLARE cursor_job_group_id INT;
  DECLARE done BOOLEAN DEFAULT FALSE;
  DECLARE total_jobs_in_job_group INT;
  DECLARE cur_n_completed INT;

  DECLARE job_group_cursor CURSOR FOR
  SELECT parent_id
  FROM job_group_parents
  WHERE batch_id = in_batch_id AND job_group_id = in_job_group_id
  ORDER BY job_group_id ASC;

  DECLARE CONTINUE HANDLER FOR NOT FOUND SET done = TRUE;

  OPEN job_group_cursor;
  update_job_group_loop: LOOP
    FETCH job_group_cursor INTO cursor_job_group_id;

    IF done THEN
      LEAVE update_job_group_loop;
    END IF;

    SELECT n_jobs INTO total_jobs_in_job_group
    FROM job_groups
    WHERE batch_id = in_batch_id AND job_group_id = cursor_job_group_id
    LOCK IN SHARE MODE;

    SELECT n_completed INTO cur_n_completed
    FROM batches_n_jobs_in_complete_states
    WHERE id = in_batch_id AND job_group_id = cursor_job_group_id
    LOCK IN SHARE MODE;

    # Grabbing an exclusive lock on job groups here could deadlock,
    # but this IF should only execute for the last job
    IF cur_n_completed = total_jobs_in_job_group THEN
      UPDATE job_groups
      SET time_completed = new_timestamp,
        `state` = 'complete'
      WHERE batch_id = in_batch_id AND job_group_id = cursor_job_group_id;
    END IF;
  END LOOP;
  CLOSE job_group_cursor;
END $$

DROP PROCEDURE IF EXISTS mark_job_complete $$
CREATE PROCEDURE mark_job_complete(
  IN in_batch_id BIGINT,
  IN in_job_id INT,
  IN in_attempt_id VARCHAR(40),
  IN in_instance_name VARCHAR(100),
  IN new_state VARCHAR(40),
  IN new_status TEXT,
  IN new_start_time BIGINT,
  IN new_end_time BIGINT,
  IN new_reason VARCHAR(40),
  IN new_timestamp BIGINT
)
BEGIN
  DECLARE cur_job_group_id INT;
  DECLARE cur_job_state VARCHAR(40);
  DECLARE cur_instance_state VARCHAR(40);
  DECLARE cur_cores_mcpu INT;
  DECLARE cur_end_time BIGINT;
  DECLARE delta_cores_mcpu INT DEFAULT 0;
  DECLARE total_jobs_in_batch INT;
  DECLARE expected_attempt_id VARCHAR(40);

  START TRANSACTION;

  SELECT n_jobs INTO total_jobs_in_batch FROM batches WHERE id = in_batch_id;

  SELECT state, cores_mcpu, job_group_id
  INTO cur_job_state, cur_cores_mcpu, cur_job_group_id
  FROM jobs
  WHERE batch_id = in_batch_id AND job_id = in_job_id
  FOR UPDATE;

  CALL add_attempt(in_batch_id, in_job_id, in_attempt_id, in_instance_name, cur_cores_mcpu, delta_cores_mcpu);

  SELECT end_time INTO cur_end_time FROM attempts
  WHERE batch_id = in_batch_id AND job_id = in_job_id AND attempt_id = in_attempt_id
  FOR UPDATE;

  UPDATE attempts
  SET start_time = new_start_time, rollup_time = new_end_time, end_time = new_end_time, reason = new_reason
  WHERE batch_id = in_batch_id AND job_id = in_job_id AND attempt_id = in_attempt_id;

  SELECT state INTO cur_instance_state FROM instances WHERE name = in_instance_name LOCK IN SHARE MODE;
  IF cur_instance_state = 'active' AND cur_end_time IS NULL THEN
    UPDATE instances_free_cores_mcpu
    SET free_cores_mcpu = free_cores_mcpu + cur_cores_mcpu
    WHERE instances_free_cores_mcpu.name = in_instance_name;

    SET delta_cores_mcpu = delta_cores_mcpu + cur_cores_mcpu;
  END IF;

  SELECT attempt_id INTO expected_attempt_id FROM jobs
  WHERE batch_id = in_batch_id AND job_id = in_job_id
  FOR UPDATE;

  IF expected_attempt_id IS NOT NULL AND expected_attempt_id != in_attempt_id THEN
    COMMIT;
    SELECT 2 as rc,
      expected_attempt_id,
      delta_cores_mcpu,
      'input attempt id does not match expected attempt id' as message;
  ELSEIF cur_job_state = 'Ready' OR cur_job_state = 'Creating' OR cur_job_state = 'Running' THEN
    UPDATE jobs
    SET state = new_state, status = new_status, attempt_id = in_attempt_id
    WHERE batch_id = in_batch_id AND job_id = in_job_id;

    # update only the record for the root job group
    # backwards compatibility for job groups that do not exist
    UPDATE batches_n_jobs_in_complete_states
      SET n_completed = (@new_n_completed := n_completed + 1),
          n_cancelled = n_cancelled + (new_state = 'Cancelled'),
          n_failed    = n_failed + (new_state = 'Error' OR new_state = 'Failed'),
          n_succeeded = n_succeeded + (new_state != 'Cancelled' AND new_state != 'Error' AND new_state != 'Failed')
      WHERE id = in_batch_id AND job_group_id = 0;

    # Grabbing an exclusive lock on batches here could deadlock,
    # but this IF should only execute for the last job
    IF @new_n_completed = total_jobs_in_batch THEN
      UPDATE batches
      SET time_completed = new_timestamp,
          `state` = 'complete'
      WHERE id = in_batch_id;
    END IF;

    # update the rest of the non-root job groups if they exist
    # necessary for backwards compatibility
    UPDATE batches_n_jobs_in_complete_states
    INNER JOIN (
      SELECT batch_id, parent_id
      FROM job_group_parents
      WHERE batch_id = in_batch_id AND job_group_id = cur_job_group_id AND job_group_id != 0
      ORDER BY job_group_id ASC
    ) AS t ON batches_n_jobs_in_complete_states.batch_id = t.batch_id AND batches_n_jobs_in_complete_states.job_group_id = t.job_group_id
    SET n_completed = n_completed + 1,
        n_cancelled = n_cancelled + (new_state = 'Cancelled'),
        n_failed = n_failed + (new_state = 'Error' OR new_state = 'Failed'),
        n_succeeded = n_succeeded + (new_state != 'Cancelled' AND new_state != 'Error' AND new_state != 'Failed');

    CALL mark_job_group_complete(in_batch_id, cur_job_group_id, new_timestamp);

    UPDATE jobs
      LEFT JOIN `jobs_telemetry` ON `jobs_telemetry`.batch_id = jobs.batch_id AND `jobs_telemetry`.job_id = jobs.job_id
      INNER JOIN `job_parents`
        ON jobs.batch_id = `job_parents`.batch_id AND
           jobs.job_id = `job_parents`.job_id
      SET jobs.state = IF(jobs.n_pending_parents = 1, 'Ready', 'Pending'),
          jobs.n_pending_parents = jobs.n_pending_parents - 1,
          jobs.cancelled = IF(new_state = 'Success', jobs.cancelled, 1),
          jobs_telemetry.time_ready = IF(jobs.n_pending_parents = 1, new_timestamp, jobs_telemetry.time_ready)
      WHERE jobs.batch_id = in_batch_id AND
            `job_parents`.batch_id = in_batch_id AND
            `job_parents`.parent_id = in_job_id;

    COMMIT;
    SELECT 0 as rc,
      cur_job_state as old_state,
      delta_cores_mcpu;
  ELSEIF cur_job_state = 'Cancelled' OR cur_job_state = 'Error' OR
         cur_job_state = 'Failed' OR cur_job_state = 'Success' THEN
    COMMIT;
    SELECT 0 as rc,
      cur_job_state as old_state,
      delta_cores_mcpu;
  ELSE
    COMMIT;
    SELECT 1 as rc,
      cur_job_state,
      delta_cores_mcpu,
      'job state not Ready, Creating, Running or complete' as message;
  END IF;
END $$

DELIMITER ;

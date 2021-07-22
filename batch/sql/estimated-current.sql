CREATE TABLE IF NOT EXISTS `globals` (
  `instance_id` VARCHAR(100) NOT NULL,
  `internal_token` VARCHAR(100) NOT NULL,
  `n_tokens` INT NOT NULL,
  `frozen` BOOLEAN NOT NULL DEFAULT FALSE
) ENGINE = InnoDB;

CREATE TABLE IF NOT EXISTS `resources` (
  `resource` VARCHAR(100) NOT NULL,
  `rate` DOUBLE NOT NULL,
  PRIMARY KEY (`resource`)
) ENGINE = InnoDB;

CREATE TABLE IF NOT EXISTS `inst_colls` (
  `name` VARCHAR(255) NOT NULL,
  `is_pool` BOOLEAN NOT NULL,
  `boot_disk_size_gb` BIGINT NOT NULL,
  `max_instances` BIGINT NOT NULL,
  `max_live_instances` BIGINT NOT NULL,
  PRIMARY KEY (`name`)
) ENGINE = InnoDB;
CREATE INDEX `inst_colls_pool` ON `inst_colls` (`pool`);

INSERT INTO inst_colls (`name`, `is_pool`, `boot_disk_size_gb`, `max_instances`, `max_live_instances`) VALUES ('standard', 1, 10, 6250, 800);
INSERT INTO inst_colls (`name`, `is_pool`, `boot_disk_size_gb`, `max_instances`, `max_live_instances`) VALUES ('highmem', 1, 10, 6250, 800);
INSERT INTO inst_colls (`name`, `is_pool`, `boot_disk_size_gb`, `max_instances`, `max_live_instances`) VALUES ('highcpu', 1, 10, 6250, 800);
INSERT INTO inst_colls (`name`, `is_pool`, `boot_disk_size_gb`, `max_instances`, `max_live_instances`) VALUES ('job-private', 0, 10, 6250, 800);

CREATE TABLE IF NOT EXISTS `pools` (
  `name` VARCHAR(255) NOT NULL,
  `worker_type` VARCHAR(100) NOT NULL,
  `worker_cores` BIGINT NOT NULL,
  `worker_local_ssd_data_disk` BOOLEAN NOT NULL DEFAULT 1,
  `worker_pd_ssd_data_disk_size_gb` BIGINT NOT NULL DEFAULT 0,
  `enable_standing_worker` BOOLEAN NOT NULL DEFAULT FALSE,
  `standing_worker_cores` BIGINT NOT NULL DEFAULT 0,
  PRIMARY KEY (`name`),
  FOREIGN KEY (`name`) REFERENCES inst_colls(name) ON DELETE CASCADE
) ENGINE = InnoDB;

INSERT INTO pools (`name`, `worker_type`, `worker_cores`, `worker_local_ssd_data_disk`,
  `worker_pd_ssd_data_disk_size_gb`, `enable_standing_worker`, `standing_worker_cores`)
VALUES ('standard', 'standard', 16, 1, 0, 1, 4);

INSERT INTO pools (`name`, `worker_type`, `worker_cores`, `worker_local_ssd_data_disk`,
  `worker_pd_ssd_data_disk_size_gb`, `enable_standing_worker`, `standing_worker_cores`)
VALUES ('highmem', 'highmem', 16, 10, 1, 0, 0, 4);

INSERT INTO pools (`name`, `worker_type`, `worker_cores`, `worker_local_ssd_data_disk`,
  `worker_pd_ssd_data_disk_size_gb`, `enable_standing_worker`, `standing_worker_cores`)
VALUES ('highcpu', 'highcpu', 16, 10, 1, 0, 0, 4);

CREATE TABLE IF NOT EXISTS `billing_projects` (
  `name` VARCHAR(100) NOT NULL,
  `status` ENUM('open', 'closed', 'deleted') NOT NULL DEFAULT 'open',
  `limit` DOUBLE DEFAULT NULL,
  `msec_mcpu` BIGINT DEFAULT 0
  PRIMARY KEY (`name`)
) ENGINE = InnoDB;
CREATE INDEX `billing_project_status` ON `billing_projects` (`status`);

CREATE TABLE IF NOT EXISTS `billing_project_users` (
  `billing_project` VARCHAR(100) NOT NULL,
  `user` VARCHAR(100) NOT NULL,
  PRIMARY KEY (`billing_project`, `user`),
  FOREIGN KEY (`billing_project`) REFERENCES billing_projects(name) ON DELETE CASCADE
) ENGINE = InnoDB;

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
  `free_cores_mcpu` INT NOT NULL,
  `time_created` BIGINT NOT NULL,
  `failed_request_count` INT NOT NULL DEFAULT 0,
  `last_updated` BIGINT NOT NULL,
  `ip_address` VARCHAR(100),
  `time_activated` BIGINT,
  `time_deactivated` BIGINT,
  `removed` BOOLEAN NOT NULL DEFAULT FALSE,
  `version` INT NOT NULL,
  `inst_coll` VARCHAR(255) NOT NULL,
  `machine_type` VARCHAR(255) NOT NULL,
  `preemptible` BOOLEAN NOT NULL,
  `config` MEDIUMTEXT,
  PRIMARY KEY (`name`),
  FOREIGN KEY (`inst_coll`) REFERENCES inst_colls(`name`)
) ENGINE = InnoDB;
CREATE INDEX `instances_removed` ON `instances` (`removed`);
CREATE INDEX `instances_inst_coll` ON `instances` (`inst_coll`);
CREATE INDEX `instances_removed_inst_coll` ON `instances` (`removed`, `inst_coll`);
CREATE INDEX `instances_time_activated` ON `instances` (`time_activated`);

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
  `cancelled` BOOLEAN NOT NULL DEFAULT FALSE,
  `n_jobs` INT NOT NULL,
  `n_completed` INT NOT NULL DEFAULT 0,
  `n_succeeded` INT NOT NULL DEFAULT 0,
  `n_failed` INT NOT NULL DEFAULT 0,
  `n_cancelled` INT NOT NULL DEFAULT 0,
  `time_created` BIGINT NOT NULL,
  `time_closed` BIGINT,
  `time_completed` BIGINT,
  `msec_mcpu` BIGINT NOT NULL DEFAULT 0,
  `token` VARCHAR(100) DEFAULT NULL,
  `format_version` INT NOT NULL,
  `cancel_after_n_failures` INT DEFAULT NULL,
  PRIMARY KEY (`id`),
  FOREIGN KEY (`billing_project`) REFERENCES billing_projects(name)
) ENGINE = InnoDB;
CREATE INDEX `batches_user_state_cancelled` ON `batches` (`user`, `state`, `cancelled`);
CREATE INDEX `batches_deleted` ON `batches` (`deleted`);
CREATE INDEX `batches_token` ON `batches` (`token`);
CREATE INDEX `batches_time_completed` ON `batches` (`time_completed`);
CREATE INDEX `batches_billing_project_state` ON `batches` (`billing_project`, `state`);

CREATE TABLE IF NOT EXISTS `batches_inst_coll_staging` (
  `batch_id` BIGINT NOT NULL,
  `inst_coll` VARCHAR(255),
  `token` INT NOT NULL,
  `n_jobs` INT NOT NULL DEFAULT 0,
  `n_ready_jobs` INT NOT NULL DEFAULT 0,
  `ready_cores_mcpu` BIGINT NOT NULL DEFAULT 0,
  PRIMARY KEY (`batch_id`, `inst_coll`, `token`),
  FOREIGN KEY (`batch_id`) REFERENCES batches(`id`) ON DELETE CASCADE,
  FOREIGN KEY (`inst_coll`) REFERENCES inst_colls(name) ON DELETE CASCADE
) ENGINE = InnoDB;
CREATE INDEX `batches_inst_coll_staging_inst_coll` ON `batches_inst_coll_staging` (`inst_coll`);

CREATE TABLE `batch_inst_coll_cancellable_resources` (
  `batch_id` BIGINT NOT NULL,
  `inst_coll` VARCHAR(255),
  `token` INT NOT NULL,
  # neither run_always nor cancelled
  `n_ready_cancellable_jobs` INT NOT NULL DEFAULT 0,
  `ready_cancellable_cores_mcpu` BIGINT NOT NULL DEFAULT 0,
  `n_creating_cancellable_jobs` INT NOT NULL DEFAULT 0,
  `n_running_cancellable_jobs` INT NOT NULL DEFAULT 0,
  `running_cancellable_cores_mcpu` BIGINT NOT NULL DEFAULT 0,
  PRIMARY KEY (`batch_id`, `inst_coll`, `token`),
  FOREIGN KEY (`batch_id`) REFERENCES batches(id) ON DELETE CASCADE,
  FOREIGN KEY (`inst_coll`) REFERENCES inst_colls(name) ON DELETE CASCADE
) ENGINE = InnoDB;
CREATE INDEX `batch_inst_coll_cancellable_resources_inst_coll` ON `batch_inst_coll_cancellable_resources` (`inst_coll`);

CREATE TABLE IF NOT EXISTS `jobs` (
  `batch_id` BIGINT NOT NULL,
  `job_id` INT NOT NULL,
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
  PRIMARY KEY (`batch_id`, `job_id`),
  FOREIGN KEY (`batch_id`) REFERENCES batches(id) ON DELETE CASCADE,
  FOREIGN KEY (`inst_coll`) REFERENCES inst_colls(name) ON DELETE CASCADE
) ENGINE = InnoDB;
CREATE INDEX `jobs_batch_id_state_always_run_inst_coll_cancelled` ON `jobs` (`batch_id`, `state`, `always_run`, `inst_coll`, `cancelled`);
CREATE INDEX `jobs_batch_id_state_always_run_cancelled` ON `jobs` (`batch_id`, `state`, `always_run`, `cancelled`);

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

CREATE TABLE IF NOT EXISTS `batch_attributes` (
  `batch_id` BIGINT NOT NULL,
  `key` VARCHAR(100) NOT NULL,
  `value` TEXT,
  PRIMARY KEY (`batch_id`, `key`),
  FOREIGN KEY (`batch_id`) REFERENCES batches(id) ON DELETE CASCADE
) ENGINE = InnoDB;
CREATE INDEX batch_attributes_key_value ON `batch_attributes` (`key`, `value`(256));

CREATE TABLE IF NOT EXISTS `aggregated_billing_project_resources` (
  `billing_project` VARCHAR(100) NOT NULL,
  `resource` VARCHAR(100) NOT NULL,
  `usage` BIGINT NOT NULL DEFAULT 0,
  PRIMARY KEY (`billing_project`, `resource`),
  FOREIGN KEY (`billing_project`) REFERENCES billing_projects(name) ON DELETE CASCADE,
  FOREIGN KEY (`resource`) REFERENCES resources(`resource`) ON DELETE CASCADE
) ENGINE = InnoDB;

CREATE TABLE IF NOT EXISTS `aggregated_batch_resources` (
  `batch_id` BIGINT NOT NULL,
  `resource` VARCHAR(100) NOT NULL,
  `token` INT NOT NULL,
  `usage` BIGINT NOT NULL DEFAULT 0,
  PRIMARY KEY (`batch_id`, `resource`, `token`),
  FOREIGN KEY (`batch_id`) REFERENCES batches(`id`) ON DELETE CASCADE,
  FOREIGN KEY (`resource`) REFERENCES resources(`resource`) ON DELETE CASCADE
) ENGINE = InnoDB;

CREATE TABLE IF NOT EXISTS `aggregated_job_resources` (
  `batch_id` BIGINT NOT NULL,
  `job_id` INT NOT NULL,
  `resource` VARCHAR(100) NOT NULL,
  `usage` BIGINT NOT NULL DEFAULT 0,
  PRIMARY KEY (`batch_id`, `job_id`, `resource`),
  FOREIGN KEY (`batch_id`) REFERENCES batches(`id`) ON DELETE CASCADE,
  FOREIGN KEY (`batch_id`, `job_id`) REFERENCES jobs(`batch_id`, `job_id`) ON DELETE CASCADE,
  FOREIGN KEY (`resource`) REFERENCES resources(`resource`) ON DELETE CASCADE
) ENGINE = InnoDB;

CREATE TABLE IF NOT EXISTS `attempt_resources` (
  `batch_id` BIGINT NOT NULL,
  `job_id` INT NOT NULL,
  `attempt_id` VARCHAR(40) NOT NULL,
  `resource` VARCHAR(100) NOT NULL,
  `quantity` BIGINT NOT NULL,
  PRIMARY KEY (`batch_id`, `job_id`, `attempt_id`, `resource`),
  FOREIGN KEY (`batch_id`) REFERENCES batches(`id`) ON DELETE CASCADE,
  FOREIGN KEY (`batch_id`, `job_id`) REFERENCES jobs(`batch_id`, `job_id`) ON DELETE CASCADE,
  FOREIGN KEY (`batch_id`, `job_id`, `attempt_id`) REFERENCES attempts(`batch_id`, `job_id`, `attempt_id`) ON DELETE CASCADE,
  FOREIGN KEY (`resource`) REFERENCES resources(`resource`) ON DELETE CASCADE
) ENGINE = InnoDB;

DELIMITER $$

DROP TRIGGER IF EXISTS instances_before_update;
CREATE TRIGGER instances_before_update BEFORE UPDATE on instances
FOR EACH ROW
BEGIN
  IF OLD.time_deactivated IS NOT NULL AND (NEW.time_deactivated IS NULL OR NEW.time_deactivated > OLD.time_deactivated) THEN
    SET NEW.time_deactivated = OLD.time_deactivated;
  END IF;
END $$

DROP TRIGGER IF EXISTS attempts_before_update;
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
END $$

DROP TRIGGER IF EXISTS attempts_after_update $$
CREATE TRIGGER attempts_after_update AFTER UPDATE ON attempts
FOR EACH ROW
BEGIN
  DECLARE job_cores_mcpu INT;
  DECLARE cur_billing_project VARCHAR(100);
  DECLARE msec_diff BIGINT;
  DECLARE cur_n_tokens INT;
  DECLARE rand_token INT;

  SELECT n_tokens INTO cur_n_tokens FROM globals LOCK IN SHARE MODE;
  SET rand_token = FLOOR(RAND() * cur_n_tokens);

  SELECT cores_mcpu INTO job_cores_mcpu FROM jobs
  WHERE batch_id = NEW.batch_id AND job_id = NEW.job_id;

  SELECT billing_project INTO cur_billing_project FROM batches WHERE id = NEW.batch_id;

  SET msec_diff = (GREATEST(COALESCE(NEW.end_time - NEW.start_time, 0), 0) -
                   GREATEST(COALESCE(OLD.end_time - OLD.start_time, 0), 0));

  INSERT INTO aggregated_billing_project_resources (billing_project, resource, token, `usage`)
  SELECT billing_project, resource, rand_token, msec_diff * quantity
  FROM attempt_resources
  JOIN batches ON batches.id = attempt_resources.batch_id
  WHERE batch_id = NEW.batch_id AND job_id = NEW.job_id AND attempt_id = NEW.attempt_id
  ON DUPLICATE KEY UPDATE `usage` = `usage` + msec_diff * quantity;

  INSERT INTO aggregated_batch_resources (batch_id, resource, token, `usage`)
  SELECT batch_id, resource, rand_token, msec_diff * quantity
  FROM attempt_resources
  WHERE batch_id = NEW.batch_id AND job_id = NEW.job_id AND attempt_id = NEW.attempt_id
  ON DUPLICATE KEY UPDATE `usage` = `usage` + msec_diff * quantity;

  INSERT INTO aggregated_job_resources (batch_id, job_id, resource, `usage`)
  SELECT batch_id, job_id, resource, msec_diff * quantity
  FROM attempt_resources
  WHERE batch_id = NEW.batch_id AND job_id = NEW.job_id AND attempt_id = NEW.attempt_id
  ON DUPLICATE KEY UPDATE `usage` = `usage` + msec_diff * quantity;
END $$

DROP TRIGGER IF EXISTS jobs_after_update $$
CREATE TRIGGER jobs_after_update AFTER UPDATE ON jobs
FOR EACH ROW
BEGIN
  DECLARE cur_user VARCHAR(100);
  DECLARE cur_batch_cancelled BOOLEAN;
  DECLARE cur_n_tokens INT;
  DECLARE rand_token INT;

  SELECT user, cancelled INTO cur_user, cur_batch_cancelled FROM batches
  WHERE id = NEW.batch_id
  LOCK IN SHARE MODE;

  SELECT n_tokens INTO cur_n_tokens FROM globals LOCK IN SHARE MODE;
  SET rand_token = FLOOR(RAND() * cur_n_tokens);

  IF OLD.state = 'Ready' THEN
    IF NOT (OLD.always_run OR OLD.cancelled OR cur_batch_cancelled) THEN
      # cancellable
      INSERT INTO batch_inst_coll_cancellable_resources (batch_id, inst_coll, token, n_ready_cancellable_jobs, ready_cancellable_cores_mcpu)
      VALUES (OLD.batch_id, OLD.inst_coll, rand_token, -1, -OLD.cores_mcpu)
      ON DUPLICATE KEY UPDATE
        n_ready_cancellable_jobs = n_ready_cancellable_jobs - 1,
        ready_cancellable_cores_mcpu = ready_cancellable_cores_mcpu - OLD.cores_mcpu;
    END IF;

    IF NOT OLD.always_run AND (OLD.cancelled OR cur_batch_cancelled) THEN
      # cancelled
      INSERT INTO user_inst_coll_resources (user, inst_coll, token, n_cancelled_ready_jobs)
      VALUES (cur_user, OLD.inst_coll, rand_token, -1)
      ON DUPLICATE KEY UPDATE
        n_cancelled_ready_jobs = n_cancelled_ready_jobs - 1;
    ELSE
      # runnable
      INSERT INTO user_inst_coll_resources (user, inst_coll, token, n_ready_jobs, ready_cores_mcpu)
      VALUES (cur_user, OLD.inst_coll, rand_token, -1, -OLD.cores_mcpu)
      ON DUPLICATE KEY UPDATE
        n_ready_jobs = n_ready_jobs - 1,
        ready_cores_mcpu = ready_cores_mcpu - OLD.cores_mcpu;
    END IF;
  ELSEIF OLD.state = 'Running' THEN
    IF NOT (OLD.always_run OR cur_batch_cancelled) THEN
      # cancellable
      INSERT INTO batch_inst_coll_cancellable_resources (batch_id, inst_coll, token, n_running_cancellable_jobs, running_cancellable_cores_mcpu)
      VALUES (OLD.batch_id, OLD.inst_coll, rand_token, -1, -OLD.cores_mcpu)
      ON DUPLICATE KEY UPDATE
        n_running_cancellable_jobs = n_running_cancellable_jobs - 1,
        running_cancellable_cores_mcpu = running_cancellable_cores_mcpu - OLD.cores_mcpu;
    END IF;

    # state = 'Running' jobs cannot be cancelled at the job level
    IF NOT OLD.always_run AND cur_batch_cancelled THEN
      # cancelled
      INSERT INTO user_inst_coll_resources (user, inst_coll, token, n_cancelled_running_jobs)
      VALUES (cur_user, OLD.inst_coll, rand_token, -1)
      ON DUPLICATE KEY UPDATE
        n_cancelled_running_jobs = n_cancelled_running_jobs - 1;
    ELSE
      # running
      INSERT INTO user_inst_coll_resources (user, inst_coll, token, n_running_jobs, running_cores_mcpu)
      VALUES (cur_user, OLD.inst_coll, rand_token, -1, -OLD.cores_mcpu)
      ON DUPLICATE KEY UPDATE
        n_running_jobs = n_running_jobs - 1,
        running_cores_mcpu = running_cores_mcpu - OLD.cores_mcpu;
    END IF;
  ELSEIF OLD.state = 'Creating' THEN
    IF NOT (OLD.always_run OR cur_batch_cancelled) THEN
      # cancellable
      INSERT INTO batch_inst_coll_cancellable_resources (batch_id, inst_coll, token, n_creating_cancellable_jobs)
      VALUES (OLD.batch_id, OLD.inst_coll, rand_token, -1)
      ON DUPLICATE KEY UPDATE
        n_creating_cancellable_jobs = n_creating_cancellable_jobs - 1;
    END IF;

    # state = 'Creating' jobs cannot be cancelled at the job level
    IF NOT OLD.always_run AND cur_batch_cancelled THEN
      # cancelled
      INSERT INTO user_inst_coll_resources (user, inst_coll, token, n_cancelled_creating_jobs)
      VALUES (cur_user, OLD.inst_coll, rand_token, -1)
      ON DUPLICATE KEY UPDATE
        n_cancelled_creating_jobs = n_creating_creating_jobs - 1;
    ELSE
      # creating
      INSERT INTO user_inst_coll_resources (user, inst_coll, token, n_creating_jobs)
      VALUES (cur_user, OLD.inst_coll, rand_token, -1)
      ON DUPLICATE KEY UPDATE
        n_creating_jobs = n_creating_jobs - 1;
    END IF;

  END IF;

  IF NEW.state = 'Ready' THEN
    IF NOT (NEW.always_run OR NEW.cancelled OR cur_batch_cancelled) THEN
      # cancellable
      INSERT INTO batch_inst_coll_cancellable_resources (batch_id, inst_coll, token, n_ready_cancellable_jobs, ready_cancellable_cores_mcpu)
      VALUES (NEW.batch_id, NEW.inst_coll, rand_token, 1, NEW.cores_mcpu)
      ON DUPLICATE KEY UPDATE
        n_ready_cancellable_jobs = n_ready_cancellable_jobs + 1,
        ready_cancellable_cores_mcpu = ready_cancellable_cores_mcpu + NEW.cores_mcpu;
    END IF;

    IF NOT NEW.always_run AND (NEW.cancelled OR cur_batch_cancelled) THEN
      # cancelled
      INSERT INTO user_inst_coll_resources (user, inst_coll, token, n_cancelled_ready_jobs)
      VALUES (cur_user, NEW.inst_coll, rand_token, 1)
      ON DUPLICATE KEY UPDATE
        n_cancelled_ready_jobs = n_cancelled_ready_jobs + 1;
    ELSE
      # runnable
      INSERT INTO user_inst_coll_resources (user, inst_coll, token, n_ready_jobs, ready_cores_mcpu)
      VALUES (cur_user, NEW.inst_coll, rand_token, 1, NEW.cores_mcpu)
      ON DUPLICATE KEY UPDATE
        n_ready_jobs = n_ready_jobs + 1,
        ready_cores_mcpu = ready_cores_mcpu + NEW.cores_mcpu;
    END IF;
  ELSEIF NEW.state = 'Running' THEN
    IF NOT (NEW.always_run OR cur_batch_cancelled) THEN
      # cancellable
      INSERT INTO batch_inst_coll_cancellable_resources (batch_id, inst_coll, token, n_running_cancellable_jobs, running_cancellable_cores_mcpu)
      VALUES (NEW.batch_id, NEW.inst_coll, rand_token, 1, NEW.cores_mcpu)
      ON DUPLICATE KEY UPDATE
        n_running_cancellable_jobs = n_running_cancellable_jobs + 1,
        running_cancellable_cores_mcpu = running_cancellable_cores_mcpu + NEW.cores_mcpu;
    END IF;

    # state = 'Running' jobs cannot be cancelled at the job level
    IF NOT NEW.always_run AND cur_batch_cancelled THEN
      # cancelled
      INSERT INTO user_inst_coll_resources (user, inst_coll, token, n_cancelled_running_jobs)
      VALUES (cur_user, NEW.inst_coll, rand_token, 1)
      ON DUPLICATE KEY UPDATE
        n_cancelled_running_jobs = n_cancelled_running_jobs + 1;
    ELSE
      # running
      INSERT INTO user_inst_coll_resources (user, inst_coll, token, n_running_jobs, running_cores_mcpu)
      VALUES (cur_user, NEW.inst_coll, rand_token, 1, NEW.cores_mcpu)
      ON DUPLICATE KEY UPDATE
        n_running_jobs = n_running_jobs + 1,
        running_cores_mcpu = running_cores_mcpu + NEW.cores_mcpu;
    END IF;
  ELSEIF NEW.state = 'Creating' THEN
    IF NOT (NEW.always_run OR cur_batch_cancelled) THEN
      # cancellable
      INSERT INTO batch_inst_coll_cancellable_resources (batch_id, inst_coll, token, n_creating_cancellable_jobs)
      VALUES (NEW.batch_id, NEW.inst_coll, rand_token, 1)
      ON DUPLICATE KEY UPDATE
        n_creating_cancellable_jobs = n_creating_cancellable_jobs + 1;
    END IF;

    # state = 'Creating' jobs cannot be cancelled at the job level
    IF NOT NEW.always_run AND cur_batch_cancelled THEN
      # cancelled
      INSERT INTO user_inst_coll_resources (user, inst_coll, token, n_cancelled_creating_jobs)
      VALUES (cur_user, NEW.inst_coll, rand_token, 1)
      ON DUPLICATE KEY UPDATE
        n_cancelled_creating_jobs = n_cancelled_creating_jobs + 1;
    ELSE
      # creating
      INSERT INTO user_inst_coll_resources (user, inst_coll, token, n_creating_jobs)
      VALUES (cur_user, NEW.inst_coll, rand_token, 1)
      ON DUPLICATE KEY UPDATE
        n_creating_jobs = n_creating_jobs + 1;
    END IF;
  END IF;
END $$

DROP TRIGGER IF EXISTS attempt_resources_after_insert $$
CREATE TRIGGER attempt_resources_after_insert AFTER INSERT ON attempt_resources
FOR EACH ROW
BEGIN
  DECLARE cur_start_time BIGINT;
  DECLARE cur_end_time BIGINT;
  DECLARE cur_billing_project VARCHAR(100);
  DECLARE msec_diff BIGINT;
  DECLARE cur_n_tokens INT;
  DECLARE rand_token INT;

  SELECT n_tokens INTO cur_n_tokens FROM globals LOCK IN SHARE MODE;
  SET rand_token = FLOOR(RAND() * cur_n_tokens);

  SELECT billing_project INTO cur_billing_project FROM batches WHERE id = NEW.batch_id;

  SELECT start_time, end_time INTO cur_start_time, cur_end_time
  FROM attempts
  WHERE batch_id = NEW.batch_id AND job_id = NEW.job_id AND attempt_id = NEW.attempt_id
  LOCK IN SHARE MODE;

  SET msec_diff = GREATEST(COALESCE(cur_end_time - cur_start_time, 0), 0);

  INSERT INTO aggregated_job_resources (batch_id, job_id, resource, `usage`)
  VALUES (NEW.batch_id, NEW.job_id, NEW.resource, NEW.quantity * msec_diff)
  ON DUPLICATE KEY UPDATE
    `usage` = `usage` + NEW.quantity * msec_diff;

  INSERT INTO aggregated_batch_resources (batch_id, resource, token, `usage`)
  VALUES (NEW.batch_id, NEW.resource, rand_token, NEW.quantity * msec_diff)
  ON DUPLICATE KEY UPDATE
    `usage` = `usage` + NEW.quantity * msec_diff;

  INSERT INTO aggregated_billing_project_resources (billing_project, resource, token, `usage`)
  VALUES (cur_billing_project, NEW.resource, rand_token, NEW.quantity * msec_diff)
  ON DUPLICATE KEY UPDATE
    `usage` = `usage` + NEW.quantity * msec_diff;
END $$

DROP PROCEDURE IF EXISTS recompute_incremental $$
CREATE PROCEDURE recompute_incremental(
) BEGIN

  DELETE FROM batches_inst_coll_staging;
  DELETE FROM batch_inst_coll_cancellable_resources;
  DELETE FROM user_inst_coll_resources;

  DROP TEMPORARY TABLE IF EXISTS `tmp_batch_inst_coll_resources`;

  CREATE TEMPORARY TABLE `tmp_batch_inst_coll_resources` AS (
    SELECT batch_id, batch_state, batch_cancelled, user, job_inst_coll,
      COALESCE(SUM(1), 0) as n_jobs,
      COALESCE(SUM(job_state = 'Ready' AND cancellable), 0) as n_ready_cancellable_jobs,
      COALESCE(SUM(IF(job_state = 'Ready' AND cancellable, cores_mcpu, 0)), 0) as ready_cancellable_cores_mcpu,
      COALESCE(SUM(job_state = 'Running' AND cancellable), 0) as n_running_cancellable_jobs,
      COALESCE(SUM(IF(job_state = 'Running' AND cancellable, cores_mcpu, 0)), 0) as running_cancellable_cores_mcpu,
      COALESCE(SUM(job_state = 'Creating' AND cancellable), 0) as n_creating_cancellable_jobs,
      COALESCE(SUM(job_state = 'Running' AND NOT cancelled), 0) as n_running_jobs,
      COALESCE(SUM(IF(job_state = 'Running' AND NOT cancelled, cores_mcpu, 0)), 0) as running_cores_mcpu,
      COALESCE(SUM(job_state = 'Ready' AND runnable), 0) as n_ready_jobs,
      COALESCE(SUM(IF(job_state = 'Ready' AND runnable, cores_mcpu, 0)), 0) as ready_cores_mcpu,
      COALESCE(SUM(job_state = 'Creating' AND NOT cancelled), 0) as n_creating_jobs,
      COALESCE(SUM(job_state = 'Ready' AND cancelled), 0) as n_cancelled_ready_jobs,
      COALESCE(SUM(job_state = 'Running' AND cancelled), 0) as n_cancelled_running_jobs,
      COALESCE(SUM(job_state = 'Creating' AND cancelled), 0) as n_cancelled_creating_jobs
    FROM (
      SELECT batches.user,
        batches.id as batch_id,
        batches.state as batch_state,
        batches.cancelled as batch_cancelled,
        jobs.inst_coll as job_inst_coll,
        jobs.state as job_state,
        jobs.cores_mcpu,
        NOT (jobs.always_run OR jobs.cancelled OR batches.cancelled) AS cancellable,
        (jobs.always_run OR NOT (jobs.cancelled OR batches.cancelled)) AS runnable,
        (NOT jobs.always_run AND (jobs.cancelled OR batches.cancelled)) AS cancelled
      FROM jobs
      INNER JOIN batches
        ON batches.id = jobs.batch_id
      LOCK IN SHARE MODE) as t
    GROUP BY batch_id, batch_state, batch_cancelled, user, job_inst_coll
  );

  INSERT INTO batches_inst_coll_staging (batch_id, inst_coll, token, n_jobs, n_ready_jobs, ready_cores_mcpu)
  SELECT batch_id, job_inst_coll, 0, n_jobs, n_ready_jobs, ready_cores_mcpu
  FROM tmp_batch_inst_coll_resources
  WHERE batch_state = 'open';

  INSERT INTO batch_inst_coll_cancellable_resources (batch_id, inst_coll, token, n_ready_cancellable_jobs,
    ready_cancellable_cores_mcpu, n_running_cancellable_jobs, running_cancellable_cores_mcpu, n_creating_cancellable_jobs)
  SELECT batch_id, job_inst_coll, 0, n_ready_cancellable_jobs, ready_cancellable_cores_mcpu,
    n_running_cancellable_jobs, running_cancellable_cores_mcpu, n_creating_cancellable_jobs
  FROM tmp_batch_inst_coll_resources
  WHERE NOT batch_cancelled;

  INSERT INTO user_inst_coll_resources (user, inst_coll, token, n_ready_jobs, ready_cores_mcpu,
    n_running_jobs, running_cores_mcpu, n_creating_jobs,
    n_cancelled_ready_jobs, n_cancelled_running_jobs, n_cancelled_creating_jobs)
  SELECT t.user, t.job_inst_coll, 0, t.n_ready_jobs, t.ready_cores_mcpu,
    t.n_running_jobs, t.running_cores_mcpu, t.n_creating_jobs,
    t.n_cancelled_ready_jobs, t.n_cancelled_running_jobs, t.n_cancelled_creating_jobs
  FROM (SELECT user, job_inst_coll,
      COALESCE(SUM(n_running_jobs), 0) as n_running_jobs,
      COALESCE(SUM(running_cores_mcpu), 0) as running_cores_mcpu,
      COALESCE(SUM(n_ready_jobs), 0) as n_ready_jobs,
      COALESCE(SUM(ready_cores_mcpu), 0) as ready_cores_mcpu,
      COALESCE(SUM(n_creating_jobs), 0) as n_creating_jobs,
      COALESCE(SUM(n_cancelled_ready_jobs), 0) as n_cancelled_ready_jobs,
      COALESCE(SUM(n_cancelled_running_jobs), 0) as n_cancelled_running_jobs,
      COALESCE(SUM(n_cancelled_creating_jobs), 0) as n_cancelled_creating_jobs
    FROM tmp_batch_inst_coll_resources
    WHERE batch_state != 'open'
    GROUP by user, job_inst_coll) as t;

  DROP TEMPORARY TABLE IF EXISTS `tmp_batch_inst_coll_resources`;

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
  SET end_time = in_timestamp, reason = in_reason
  WHERE instance_name = in_instance_name;

  IF cur_state = 'pending' or cur_state = 'active' THEN
    UPDATE jobs
    INNER JOIN attempts ON jobs.batch_id = attempts.batch_id AND jobs.job_id = attempts.job_id AND jobs.attempt_id = attempts.attempt_id
    SET state = 'Ready',
        jobs.attempt_id = NULL
    WHERE instance_name = in_instance_name AND (state = 'Running' OR state = 'Creating');

    UPDATE instances SET state = 'inactive', free_cores_mcpu = cores_mcpu WHERE name = in_instance_name;

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

DROP PROCEDURE IF EXISTS close_batch $$
CREATE PROCEDURE close_batch(
  IN in_batch_id BIGINT,
  IN in_timestamp BIGINT
)
BEGIN
  DECLARE cur_batch_state VARCHAR(40);
  DECLARE expected_n_jobs INT;
  DECLARE staging_n_jobs INT;
  DECLARE staging_n_ready_jobs INT;
  DECLARE staging_ready_cores_mcpu BIGINT;
  DECLARE cur_user VARCHAR(100);

  START TRANSACTION;

  SELECT `state`, n_jobs INTO cur_batch_state, expected_n_jobs FROM batches
  WHERE id = in_batch_id AND NOT deleted
  FOR UPDATE;

  IF cur_batch_state != 'open' THEN
    COMMIT;
    SELECT 0 as rc;
  ELSE
    SELECT COALESCE(SUM(n_jobs), 0), COALESCE(SUM(n_ready_jobs), 0), COALESCE(SUM(ready_cores_mcpu), 0)
    INTO staging_n_jobs, staging_n_ready_jobs, staging_ready_cores_mcpu
    FROM batches_inst_coll_staging
    WHERE batch_id = in_batch_id
    FOR UPDATE;

    SELECT user INTO cur_user FROM batches WHERE id = in_batch_id;

    IF staging_n_jobs = expected_n_jobs THEN
      IF expected_n_jobs = 0 THEN
        UPDATE batches SET `state` = 'complete', time_completed = in_timestamp, time_closed = in_timestamp
          WHERE id = in_batch_id;
      ELSE
        UPDATE batches SET `state` = 'running', time_closed = in_timestamp
          WHERE id = in_batch_id;
      END IF;

      INSERT INTO user_inst_coll_resources (user, inst_coll, token, n_ready_jobs, ready_cores_mcpu)
      SELECT user, inst_coll, 0, @n_ready_jobs := COALESCE(SUM(n_ready_jobs), 0), @ready_cores_mcpu := COALESCE(SUM(ready_cores_mcpu), 0)
      FROM batches_inst_coll_staging
      JOIN batches ON batches.id = batches_inst_coll_staging.batch_id
      WHERE batch_id = in_batch_id
      GROUP BY `user`, inst_coll
      ON DUPLICATE KEY UPDATE
        n_ready_jobs = n_ready_jobs + @n_ready_jobs,
        ready_cores_mcpu = ready_cores_mcpu + @ready_cores_mcpu;

      DELETE FROM batches_inst_coll_staging WHERE batch_id = in_batch_id;

      COMMIT;
      SELECT 0 as rc;
    ELSE
      ROLLBACK;
      SELECT 2 as rc, expected_n_jobs, staging_n_jobs as actual_n_jobs, 'wrong number of jobs' as message;
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

  SELECT user, `state`, cancelled INTO cur_user, cur_batch_state, cur_cancelled FROM batches
  WHERE id = in_batch_id
  FOR UPDATE;

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
    WHERE batch_id = in_batch_id
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

    UPDATE batches SET cancelled = 1 WHERE id = in_batch_id;
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
  DECLARE attempt_exists BOOLEAN;
  DECLARE cur_instance_state VARCHAR(40);
  SET delta_cores_mcpu = IFNULL(delta_cores_mcpu, 0);

  SET attempt_exists = EXISTS (SELECT * FROM attempts
                               WHERE batch_id = in_batch_id AND
                                 job_id = in_job_id AND attempt_id = in_attempt_id
                               FOR UPDATE);

  IF NOT attempt_exists AND in_attempt_id IS NOT NULL THEN
    INSERT INTO attempts (batch_id, job_id, attempt_id, instance_name) VALUES (in_batch_id, in_job_id, in_attempt_id, in_instance_name);
    SELECT state INTO cur_instance_state FROM instances WHERE name = in_instance_name LOCK IN SHARE MODE;
    # instance pending when attempt is from a job private instance
    IF cur_instance_state = 'pending' OR cur_instance_state = 'active' THEN
      UPDATE instances SET free_cores_mcpu = free_cores_mcpu - in_cores_mcpu WHERE name = in_instance_name;
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

  SELECT (jobs.cancelled OR batches.cancelled) AND NOT jobs.always_run
  INTO cur_job_cancel
  FROM jobs
  INNER JOIN batches ON batches.id = jobs.batch_id
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
  SET end_time = new_end_time, reason = new_reason
  WHERE batch_id = in_batch_id AND job_id = in_job_id AND attempt_id = in_attempt_id;

  SELECT state INTO cur_instance_state FROM instances WHERE name = in_instance_name LOCK IN SHARE MODE;

  IF cur_instance_state = 'active' AND cur_end_time IS NULL THEN
    UPDATE instances
    SET free_cores_mcpu = free_cores_mcpu + cur_cores_mcpu
    WHERE name = in_instance_name;

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

  SELECT (jobs.cancelled OR batches.cancelled) AND NOT jobs.always_run
  INTO cur_job_cancel
  FROM jobs
  INNER JOIN batches ON batches.id = jobs.batch_id
  WHERE batch_id = in_batch_id AND job_id = in_job_id
  LOCK IN SHARE MODE;

  CALL add_attempt(in_batch_id, in_job_id, in_attempt_id, in_instance_name, cur_cores_mcpu, delta_cores_mcpu);

  UPDATE attempts SET start_time = new_start_time
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

  SELECT (jobs.cancelled OR batches.cancelled) AND NOT jobs.always_run
  INTO cur_job_cancel
  FROM jobs
  INNER JOIN batches ON batches.id = jobs.batch_id
  WHERE batch_id = in_batch_id AND job_id = in_job_id
  LOCK IN SHARE MODE;

  CALL add_attempt(in_batch_id, in_job_id, in_attempt_id, in_instance_name, cur_cores_mcpu, delta_cores_mcpu);

  UPDATE attempts SET start_time = new_start_time
  WHERE batch_id = in_batch_id AND job_id = in_job_id AND attempt_id = in_attempt_id;

  SELECT state INTO cur_instance_state FROM instances WHERE name = in_instance_name LOCK IN SHARE MODE;

  IF cur_job_state = 'Ready' AND NOT cur_job_cancel AND cur_instance_state = 'active' THEN
    UPDATE jobs SET state = 'Running', attempt_id = in_attempt_id WHERE batch_id = in_batch_id AND job_id = in_job_id;
  END IF;

  COMMIT;
  SELECT 0 as rc, delta_cores_mcpu;
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
  DECLARE cur_job_state VARCHAR(40);
  DECLARE cur_instance_state VARCHAR(40);
  DECLARE cur_cores_mcpu INT;
  DECLARE cur_end_time BIGINT;
  DECLARE delta_cores_mcpu INT DEFAULT 0;
  DECLARE expected_attempt_id VARCHAR(40);

  START TRANSACTION;

  SELECT state, cores_mcpu
  INTO cur_job_state, cur_cores_mcpu
  FROM jobs
  WHERE batch_id = in_batch_id AND job_id = in_job_id
  FOR UPDATE;

  CALL add_attempt(in_batch_id, in_job_id, in_attempt_id, in_instance_name, cur_cores_mcpu, delta_cores_mcpu);

  SELECT end_time INTO cur_end_time FROM attempts
  WHERE batch_id = in_batch_id AND job_id = in_job_id AND attempt_id = in_attempt_id
  FOR UPDATE;

  UPDATE attempts
  SET start_time = new_start_time, end_time = new_end_time, reason = new_reason
  WHERE batch_id = in_batch_id AND job_id = in_job_id AND attempt_id = in_attempt_id;

  SELECT state INTO cur_instance_state FROM instances WHERE name = in_instance_name LOCK IN SHARE MODE;
  IF cur_instance_state = 'active' AND cur_end_time IS NULL THEN
    UPDATE instances
    SET free_cores_mcpu = free_cores_mcpu + cur_cores_mcpu
    WHERE name = in_instance_name;

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

    UPDATE batches SET n_completed = n_completed + 1 WHERE id = in_batch_id;
    UPDATE batches
      SET time_completed = new_timestamp,
          `state` = 'complete'
      WHERE id = in_batch_id AND n_completed = batches.n_jobs;

    IF new_state = 'Cancelled' THEN
      UPDATE batches SET n_cancelled = n_cancelled + 1 WHERE id = in_batch_id;
    ELSEIF new_state = 'Error' OR new_state = 'Failed' THEN
      UPDATE batches SET n_failed = n_failed + 1 WHERE id = in_batch_id;
    ELSE
      UPDATE batches SET n_succeeded = n_succeeded + 1 WHERE id = in_batch_id;
    END IF;

    UPDATE jobs
      INNER JOIN `job_parents`
        ON jobs.batch_id = `job_parents`.batch_id AND
           jobs.job_id = `job_parents`.job_id
      SET jobs.state = IF(jobs.n_pending_parents = 1, 'Ready', 'Pending'),
          jobs.n_pending_parents = jobs.n_pending_parents - 1,
          jobs.cancelled = IF(new_state = 'Success', jobs.cancelled, 1)
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

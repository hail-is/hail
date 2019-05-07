CREATE TABLE IF NOT EXISTS `jobs` (
  `id` BIGINT NOT NULL AUTO_INCREMENT,
  `state` VARCHAR(40) NOT NULL,
  `exit_code` INT,
  `batch_id` BIGINT NOT NULL,
  `pod_name` VARCHAR(1024),
  `pvc` TEXT(65535),
  `callback` TEXT(65535),
  `task_idx` INT NOT NULL,
  `always_run` BOOLEAN,
  `cancelled` BOOLEAN,
  `time_created` TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  `duration` BIGINT,
  `userdata` TEXT(65535),
  `user` VARCHAR(100),
  `attributes` TEXT(65535),
  `tasks` TEXT(65535),
  `input_log_uri` VARCHAR(1024),
  `main_log_uri` VARCHAR(1024),
  `output_log_uri` VARCHAR(1024),
  PRIMARY KEY (`id`)
) ENGINE = InnoDB;
CREATE INDEX jobs_user ON jobs (user);
CREATE INDEX jobs_batch ON jobs (batch_id);

CREATE TABLE IF NOT EXISTS `jobs-parents` (
  `job_id` BIGINT NOT NULL,
  `parent_id` BIGINT NOT NULL,
  PRIMARY KEY (`job_id`, `parent_id`)
) ENGINE = InnoDB;
CREATE INDEX jobs_parents_parent_id ON `jobs-parents` (parent_id);

CREATE TABLE IF NOT EXISTS `batch` (
  `id` BIGINT NOT NULL AUTO_INCREMENT,
  `userdata` TEXT(65535),
  `user` VARCHAR(100),
  `attributes` TEXT(65535),
  `callback` TEXT(65535),
  `ttl` INT,
  `is_open` BOOLEAN NOT NULL,
  `time_created` TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`)
) ENGINE = InnoDB;
CREATE INDEX batch_user ON batch (user);

DROP TABLE IF EXISTS `job_groups`;
CREATE TABLE IF NOT EXISTS `job_groups` (
  `batch_id` BIGINT NOT NULL,
  `job_group_id` INT NOT NULL,
  `attributes` TEXT,
  `cancel_after_n_failures` INT DEFAULT NULL,
  `path` VARCHAR(255) NOT NULL COLLATE utf8mb4_0900_as_cs,  # fix in front end max length of path
  `state` ENUM('running', 'complete') NOT NULL,
  `n_jobs` INT NOT NULL,
  `time_created` BIGINT NOT NULL,
  `time_completed` BIGINT,
  `callback` VARCHAR(255),
  `token` VARCHAR(100),
  PRIMARY KEY (`batch_id`, `job_group_id`),
  UNIQUE (`batch_id`, `path`),
  FOREIGN KEY (`batch_id`) REFERENCES batches(`id`) ON DELETE CASCADE
) ENGINE = InnoDB;
CREATE INDEX `job_groups_path` ON `job_groups` (`batch_id`, `path`);
CREATE INDEX `job_groups_state_callback` ON `job_groups` (`batch_id`, `state`, `callback`);
CREATE INDEX `job_groups_time_created` ON `job_groups` (`batch_id`, `time_created`);
CREATE INDEX `job_groups_time_completed` ON `job_groups` (`batch_id`, `time_completed`);
CREATE INDEX `job_groups_state_cancel_after_n_failures` ON `job_groups` (`batch_id`, `state`, `cancel_after_n_failures`);

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
CREATE INDEX `job_group_parents_parent_id` ON `job_group_parents` (`batch_id`, `job_group_id`, `parent_id`, `level`);
CREATE INDEX `job_group_parents_level` ON `job_group_parents` (`batch_id`, `level`);

ALTER TABLE batches ADD COLUMN migrated BOOLEAN NOT NULL DEFAULT 0, ALGORITHM=INSTANT;

ALTER TABLE jobs ADD COLUMN job_group_id INT NOT NULL DEFAULT 1, ALGORITHM=INSTANT;
CREATE INDEX jobs_batch_id_job_group_id ON `jobs` (`batch_id`, `job_group_id`);

ALTER TABLE batch_attributes ADD COLUMN job_group_id INT NOT NULL DEFAULT 1, ALGORITHM=INSTANT;
CREATE INDEX batch_attributes_batch_id_key_value ON `batch_attributes` (`batch_id`, `job_group_id`, `key`, `value`(256));
CREATE INDEX batch_attributes_batch_id_value ON `batch_attributes` (`batch_id`, `job_group_id`, `value`(256));

ALTER TABLE batches_cancelled ADD COLUMN job_group_id INT NOT NULL DEFAULT 1, ALGORITHM=INSTANT;

ALTER TABLE batches_inst_coll_staging ADD COLUMN job_group_id INT NOT NULL DEFAULT 1, ALGORITHM=INSTANT;
CREATE INDEX batches_inst_coll_staging_batch_id_jg_id ON batches_inst_coll_staging (`batch_id`, `job_group_id`);

ALTER TABLE batch_inst_coll_cancellable_resources ADD COLUMN job_group_id INT NOT NULL DEFAULT 1, ALGORITHM=INSTANT;
CREATE INDEX batch_inst_coll_cancellable_resources_jg_id ON `batch_inst_coll_cancellable_resources` (`batch_id`, `job_group_id`);

ALTER TABLE aggregated_batch_resources_v2 ADD COLUMN job_group_id INT NOT NULL DEFAULT 1, ALGORITHM=INSTANT;

ALTER TABLE aggregated_batch_resources_v3 ADD COLUMN job_group_id INT NOT NULL DEFAULT 1, ALGORITHM=INSTANT;

ALTER TABLE batches_n_jobs_in_complete_states ADD COLUMN job_group_id INT NOT NULL DEFAULT 1, ALGORITHM=INSTANT;
ALTER TABLE batches_n_jobs_in_complete_states ADD COLUMN token INT NOT NULL DEFAULT 0, ALGORITHM=INSTANT;

DELIMITER $$

DROP TRIGGER IF EXISTS batches_after_update $$
CREATE TRIGGER batches_after_update AFTER UPDATE ON batches
FOR EACH ROW
BEGIN
  IF OLD.migrated = 0 AND NEW.migrated = 1 THEN
    INSERT INTO job_groups (batch_id, job_group_id, cancel_after_n_failures, path, `state`, n_jobs, time_created, time_completed, callback, attributes)
    VALUES (NEW.id, 1, NEW.cancel_after_n_failures, '/', NEW.state, NEW.n_jobs, NEW.time_created, NEW.time_completed, NEW.callback, NEW.attributes);

    INSERT INTO job_group_parents (batch_id, job_group_id, parent_id, `level`)
    VALUES (NEW.id, 1, 1, 0);
  ELSE
    UPDATE job_groups
    SET cancel_after_n_failures = NEW.cancel_after_n_failures,
      `state` = NEW.state,
      n_jobs = NEW.n_jobs,
      time_created = NEW.time_created,
      time_completed = NEW.time_completed,
      callback = NEW.callback
    WHERE batch_id = NEW.id AND job_group_id = 1;
  END IF;
END $$

DELIMITER ;

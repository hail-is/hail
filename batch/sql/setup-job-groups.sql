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
CREATE INDEX `job_group_parents_p_id_level` ON `job_group_parents` (`batch_id`, `parent_id`, `level`);
CREATE INDEX `job_group_parents_jg_id_level` ON `job_group_parents` (`batch_id`, `job_group_id`, `level`);

ALTER TABLE batches ADD COLUMN migrated_batch BOOLEAN NOT NULL DEFAULT 0, ALGORITHM=INSTANT;

ALTER TABLE jobs ADD COLUMN job_group_id INT NOT NULL DEFAULT 0, ALGORITHM=INSTANT;
CREATE INDEX jobs_batch_id_job_group_id ON `jobs` (`batch_id`, `job_group_id`);
# we do not need an additional index for the scheduler with job groups as job group does not matter

ALTER TABLE batch_attributes ADD COLUMN job_group_id INT NOT NULL DEFAULT 0, ALGORITHM=INSTANT;
CREATE INDEX batch_attributes_batch_id_key_value ON `batch_attributes` (`batch_id`, `job_group_id`, `key`, `value`(256));
CREATE INDEX batch_attributes_batch_id_value ON `batch_attributes` (`batch_id`, `job_group_id`, `value`(256));

ALTER TABLE batches_cancelled ADD COLUMN job_group_id INT NOT NULL DEFAULT 0, ALGORITHM=INSTANT;

ALTER TABLE batches_inst_coll_staging ADD COLUMN job_group_id INT NOT NULL DEFAULT 0, ALGORITHM=INSTANT;
CREATE INDEX batches_inst_coll_staging_batch_id_jg_id ON batches_inst_coll_staging (`batch_id`, `job_group_id`);

ALTER TABLE batch_inst_coll_cancellable_resources ADD COLUMN job_group_id INT NOT NULL DEFAULT 0, ALGORITHM=INSTANT;
CREATE INDEX batch_inst_coll_cancellable_resources_jg_id ON `batch_inst_coll_cancellable_resources` (`batch_id`, `job_group_id`);

ALTER TABLE aggregated_batch_resources_v2 ADD COLUMN job_group_id INT NOT NULL DEFAULT 0, ALGORITHM=INSTANT;
ALTER TABLE aggregated_batch_resources_v3 ADD COLUMN job_group_id INT NOT NULL DEFAULT 0, ALGORITHM=INSTANT;

ALTER TABLE batches_n_jobs_in_complete_states ADD COLUMN job_group_id INT NOT NULL DEFAULT 0, ALGORITHM=INSTANT;

DELIMITER $$

DROP TRIGGER IF EXISTS batches_after_update $$
CREATE TRIGGER batches_after_update AFTER UPDATE ON batches
FOR EACH ROW
BEGIN
  IF OLD.migrated_batch = 0 AND NEW.migrated_batch = 1 THEN
    INSERT INTO job_groups (batch_id, job_group_id, `user`, cancel_after_n_failures, `state`, n_jobs, time_created, time_completed, callback, attributes)
    VALUES (NEW.id, 0, NEW.`user`, NEW.cancel_after_n_failures, NEW.state, NEW.n_jobs, NEW.time_created, NEW.time_completed, NEW.callback, NEW.attributes);

    INSERT INTO job_group_parents (batch_id, job_group_id, parent_id, `level`)
    VALUES (NEW.id, 0, 0, 0);
  ELSE
    UPDATE job_groups
    SET `user` = NEW.`user`,
      cancel_after_n_failures = NEW.cancel_after_n_failures,
      `state` = NEW.state,
      n_jobs = NEW.n_jobs,
      time_created = NEW.time_created,
      time_completed = NEW.time_completed,
      callback = NEW.callback,
      attributes = NEW.attributes
    WHERE batch_id = NEW.id AND job_group_id = 0;
  END IF;
END $$

DELIMITER ;

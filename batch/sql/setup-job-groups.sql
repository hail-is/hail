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

    INSERT INTO batches_cancelled (id) VALUES (in_batch_id);
  END IF;

  COMMIT;
END $$

DELIMITER ;

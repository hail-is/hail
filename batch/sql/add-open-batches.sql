ALTER TABLE batches ADD COLUMN time_updated BIGINT, ALGORITHM=INSTANT;

CREATE TABLE IF NOT EXISTS `batch_updates` (
  `id` BIGINT NOT NULL AUTO_INCREMENT,
  `update_id` VARCHAR(40) NOT NULL,
  `start_job_id` INT,
  `end_job_id` INT,
  `n_jobs` INT NOT NULL,
  `committed` BOOLEAN NOT NULL DEFAULT FALSE,
  `time_created` BIGINT,
  `time_committed` BIGINT,
  PRIMARY KEY (`id`, `update_id`),
  FOREIGN KEY (`id`) REFERENCES batches(id) ON DELETE CASCADE
) ENGINE = InnoDB;
CREATE INDEX `batch_updates_committed` ON `batch_updates` (`id`, `committed`);
CREATE INDEX `batch_updates_start_end_job_id` ON `batch_updates` (`id`, start_job_id, end_job_id);

CREATE TABLE IF NOT EXISTS `batch_updates_inst_coll_staging` (
  `batch_id` BIGINT NOT NULL,
  `update_id` VARCHAR(40) NOT NULL,
  `inst_coll` VARCHAR(255),
  `token` INT NOT NULL,
  `n_jobs` INT NOT NULL DEFAULT 0,
  `n_ready_jobs` INT NOT NULL DEFAULT 0,
  `ready_cores_mcpu` BIGINT NOT NULL DEFAULT 0,
  PRIMARY KEY (`batch_id`, `update_id`, `inst_coll`, `token`),
  FOREIGN KEY (`batch_id`, `update_id`) REFERENCES batch_updates(`id`, `update_id`) ON DELETE CASCADE,
  FOREIGN KEY (`inst_coll`) REFERENCES inst_colls(name) ON DELETE CASCADE
) ENGINE = InnoDB;
CREATE INDEX `batch_updates_inst_coll_staging_inst_coll` ON `batch_updates_inst_coll_staging` (`inst_coll`);

CREATE TABLE `batch_inst_coll_cancellable_resources_staging` (
  `batch_id` BIGINT NOT NULL,
  `update_id` VARCHAR(40) NOT NULL,
  `inst_coll` VARCHAR(255),
  `token` INT NOT NULL,
  # neither run_always nor cancelled
  `n_ready_cancellable_jobs` INT NOT NULL DEFAULT 0,
  `ready_cancellable_cores_mcpu` BIGINT NOT NULL DEFAULT 0,
  `n_creating_cancellable_jobs` INT NOT NULL DEFAULT 0,
  `n_running_cancellable_jobs` INT NOT NULL DEFAULT 0,
  `running_cancellable_cores_mcpu` BIGINT NOT NULL DEFAULT 0,
  PRIMARY KEY (`batch_id`, `inst_coll`, `token`),
  FOREIGN KEY (`batch_id`, `update_id`) REFERENCES batch_updates(`id`, `update_id`) ON DELETE CASCADE,
  FOREIGN KEY (`inst_coll`) REFERENCES inst_colls(name) ON DELETE CASCADE
) ENGINE = InnoDB;

DELIMITER $$

DROP PROCEDURE IF EXISTS commit_batch_update $$
CREATE PROCEDURE commit_batch_update(
  IN in_batch_id BIGINT,
  IN in_update_id VARCHAR(40),
  IN in_timestamp BIGINT
)
BEGIN
  DECLARE cur_update_committed BOOLEAN;
  DECLARE cur_update_start_job_id INT;
  DECLARE cur_update_end_job_id INT;
  DECLARE expected_n_jobs INT;
  DECLARE staging_n_jobs INT;
  DECLARE staging_n_ready_jobs INT;
  DECLARE staging_ready_cores_mcpu BIGINT;
  DECLARE cur_user VARCHAR(100);
  DECLARE cur_other_updates_in_progress INT;

  START TRANSACTION;

  SELECT committed, n_jobs, start_job_id, end_job_id INTO
    cur_update_committed, expected_n_jobs, cur_update_start_job_id, cur_update_end_job_id
  FROM batch_updates
  WHERE id = in_batch_id AND update_id = in_update_id
  FOR UPDATE;

  IF cur_update_committed THEN
    COMMIT;
    SELECT 0 as rc;
  ELSE
    SELECT SUM(NOT committed) INTO cur_other_updates_in_progress
    FROM batch_updates
    WHERE id = in_batch_id AND update_id != in_update_id
    GROUP BY id
    FOR UPDATE;

    SELECT COALESCE(SUM(n_jobs), 0), COALESCE(SUM(n_ready_jobs), 0), COALESCE(SUM(ready_cores_mcpu), 0)
    INTO staging_n_jobs, staging_n_ready_jobs, staging_ready_cores_mcpu
    FROM batch_updates_inst_coll_staging
    WHERE batch_id = in_batch_id AND update_id = in_update_id
    FOR UPDATE;

    SELECT user INTO cur_user FROM batches WHERE id = in_batch_id;

    IF staging_n_jobs = expected_n_jobs THEN
      UPDATE batches SET `state` = IF(expected_n_jobs != 0, 'running', state),
        time_updated = in_timestamp,
        time_completed = IF(expected_n_jobs != 0, NULL, time_completed)
      WHERE id = in_batch_id;

      INSERT INTO user_inst_coll_resources (user, inst_coll, token, n_ready_jobs, ready_cores_mcpu)
      SELECT user, inst_coll, 0, @n_ready_jobs := COALESCE(SUM(n_ready_jobs), 0), @ready_cores_mcpu := COALESCE(SUM(ready_cores_mcpu), 0)
      FROM batch_updates_inst_coll_staging
      JOIN batches ON batches.id = batch_updates_inst_coll_staging.batch_id
      WHERE batch_id = in_batch_id AND update_id = in_update_id
      GROUP BY `user`, inst_coll
      ON DUPLICATE KEY UPDATE
        n_ready_jobs = n_ready_jobs + @n_ready_jobs,
        ready_cores_mcpu = ready_cores_mcpu + @ready_cores_mcpu;

      INSERT INTO batch_inst_coll_cancellable_resources (batch_id, inst_coll, token, n_ready_cancellable_jobs,
        ready_cancellable_cores_mcpu, n_creating_cancellable_jobs, n_running_cancellable_jobs, running_cancellable_cores_mcpu)
      SELECT batch_id, inst_coll, 0,
        @n_ready_cancellable_jobs := COALESCE(SUM(n_ready_cancellable_jobs), 0),
        @ready_cancellable_cores_mcpu := COALESCE(SUM(ready_cancellable_cores_mcpu), 0),
        @n_creating_cancellable_jobs := COALESCE(SUM(n_creating_cancellable_jobs), 0),
        @n_running_cancellable_jobs := COALESCE(SUM(n_running_cancellable_jobs), 0),
        @running_cancellable_cores_mcpu := COALESCE(SUM(running_cancellable_cores_mcpu), 0)
      FROM batch_inst_coll_cancellable_resources_staging
      WHERE batch_id = in_batch_id AND update_id = in_update_id
      GROUP BY batch_id, inst_coll
      ON DUPLICATE KEY UPDATE
        n_ready_cancellable_jobs = n_ready_cancellable_jobs + @n_ready_cancellable_jobs,
        ready_cancellable_cores_mcpu = ready_cancellable_cores_mcpu + @ready_cancellable_cores_mcpu,
        n_creating_cancellable_jobs = n_creating_cancellable_jobs + @n_creating_cancellable_jobs,
        n_running_cancellable_jobs = n_running_cancellable_jobs + @n_running_cancellable_jobs,
        running_cancellable_cores_mcpu = running_cancellable_cores_mcpu + @running_cancellable_cores_mcpu;

      DELETE FROM batch_updates_inst_coll_staging WHERE batch_id = in_batch_id AND update_id = in_update_id;
      DELETE FROM batch_inst_coll_cancellable_resources_staging WHERE batch_id = in_batch_id AND update_id = in_update_id;

      IF cur_update_start_job_id != 1 THEN
        # FIXME see if an exists query is faster
        UPDATE jobs
          LEFT JOIN (
            SELECT `job_parents`.batch_id, `job_parents`.job_id,
              COALESCE(SUM(1), 0) AS n_parents,
              COALESCE(SUM(state IN ('Pending', 'Ready', 'Creating', 'Running')), 0) AS n_pending_parents,
              COALESCE(SUM(state = 'Success'), 0) AS n_succeeded
            FROM `job_parents`
            LEFT JOIN `jobs` ON jobs.batch_id = `job_parents`.batch_id AND jobs.job_id = `job_parents`.parent_id
            WHERE `job_parents`.batch_id = in_batch_id AND
              `job_parents`.job_id >= cur_update_start_job_id AND
              `job_parents`.job_id <= cur_update_end_job_id
            GROUP BY `job_parents`.batch_id, `job_parents`.job_id
            FOR UPDATE
          ) AS t
            ON jobs.batch_id = t.batch_id AND
               jobs.job_id = t.job_id
          SET jobs.state = IF(COALESCE(t.n_pending_parents, 0) = 0, 'Ready', 'Pending'),
              jobs.n_pending_parents = COALESCE(t.n_pending_parents, 0),
              jobs.cancelled = IF(COALESCE(t.n_succeeded, 0) = COALESCE(t.n_parents - t.n_pending_parents, 0), jobs.cancelled, 1)
          WHERE jobs.batch_id = in_batch_id AND jobs.job_id >= cur_update_start_job_id AND
              jobs.job_id <= cur_update_end_job_id;
      END IF;

      UPDATE batch_updates
      SET committed = 1, time_committed = in_timestamp
      WHERE id = in_batch_id AND update_id = in_update_id;

      COMMIT;
      SELECT 0 as rc;
    ELSE
      ROLLBACK;
      SELECT 1 as rc, expected_n_jobs, staging_n_jobs as actual_n_jobs, 'wrong number of jobs' as message;
    END IF;
  END IF;
END $$

DELIMITER ;

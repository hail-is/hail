ALTER TABLE batches ADD COLUMN `time_last_updated` BIGINT;
ALTER TABLE batches ADD COLUMN `closed` BOOLEAN NOT NULL DEFAULT FALSE;
ALTER TABLE batches ADD COLUMN `max_idle_time` INT DEFAULT NULL;
ALTER TABLE batches ADD COLUMN `n_commits` INT NOT NULL DEFAULT 0;
ALTER TABLE batches MODIFY COLUMN `n_jobs` INT NOT NULL DEFAULT 0;
CREATE INDEX `batches_closed_state_format_version_last_updated` ON `batches` (`closed`, `state`, `format_version`, `time_last_updated`);

ALTER TABLE jobs ADD COLUMN `committed` BOOLEAN NOT NULL;
CREATE INDEX `jobs_committed` ON `jobs` (`committed`);

DELIMITER $$

DROP PROCEDURE IF EXISTS close_batch $$

DROP PROCEDURE IF EXISTS cancel_batch $$
CREATE PROCEDURE cancel_batch(
  IN in_batch_id VARCHAR(100),
  IN in_timestamp BIGINT
)
BEGIN
  DECLARE cur_user VARCHAR(100);
  DECLARE cur_batch_state VARCHAR(40);
  DECLARE cur_cancelled BOOLEAN;

  START TRANSACTION;

  SELECT user, `state`, cancelled INTO cur_user, cur_batch_state, cur_cancelled
  FROM batches
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

    DELETE FROM batches_inst_coll_staging WHERE batch_id = in_batch_id;

  END IF;

  UPDATE batches
  SET cancelled = 1, closed = 1, time_completed = IF(n_completed != n_jobs, NULL, in_timestamp)
  WHERE id = in_batch_id;

  COMMIT;
END $$

DROP PROCEDURE IF EXISTS commit_staged_jobs $$
CREATE PROCEDURE commit_staged_jobs(
  IN in_batch_id BIGINT,
  IN in_timestamp BIGINT
)
BEGIN
  DECLARE cur_batch_closed BOOLEAN;
  DECLARE cur_n_commits INT;
  DECLARE staging_n_jobs INT;
  DECLARE staging_n_ready_jobs INT;
  DECLARE staging_ready_cores_mcpu BIGINT;

  START TRANSACTION;

  SELECT `closed`, `n_commits` INTO cur_batch_closed, cur_n_commits FROM batches
  WHERE id = in_batch_id AND NOT deleted
  FOR UPDATE;

  IF cur_batch_closed THEN
    COMMIT;
    SELECT 1 as rc;
  ELSE
    SELECT COALESCE(SUM(n_jobs), 0), COALESCE(SUM(n_ready_jobs), 0), COALESCE(SUM(ready_cores_mcpu), 0)
    INTO staging_n_jobs, staging_n_ready_jobs, staging_ready_cores_mcpu
    FROM batches_inst_coll_staging
    WHERE batch_id = in_batch_id
    FOR UPDATE;

    INSERT INTO user_inst_coll_resources (user, inst_coll, token, n_ready_jobs, ready_cores_mcpu)
    SELECT user, inst_coll, 0, @n_ready_jobs := COALESCE(SUM(n_ready_jobs), 0), @ready_cores_mcpu := COALESCE(SUM(ready_cores_mcpu), 0)
    FROM batches_inst_coll_staging
    JOIN batches ON batches.id = batches_inst_coll_staging.batch_id
    WHERE batch_id = in_batch_id
    GROUP BY `user`, inst_coll
    ON DUPLICATE KEY UPDATE
      n_ready_jobs = n_ready_jobs + @n_ready_jobs,
      ready_cores_mcpu = ready_cores_mcpu + @ready_cores_mcpu;

    UPDATE batches
    SET `state` = IF(staging_n_jobs != 0,
                     'running',
                     IF(n_jobs != 0, `state`, 'complete')),
      time_last_updated = in_timestamp,
      n_jobs = n_jobs + staging_n_jobs,
      n_commits = n_commits + 1
    WHERE id = in_batch_id;

    IF cur_n_commits > 0 THEN
      UPDATE jobs
        INNER JOIN (
          SELECT `job_parents`.batch_id, `job_parents`.job_id,
            COALESCE(SUM(1), 0) AS n_parents,
            COALESCE(SUM(state IN ('Pending', 'Ready', 'Creating', 'Running')), 0) AS n_pending_parents,
            COALESCE(SUM(state = 'Success'), 0) AS n_succeeded
          FROM `job_parents`
          LEFT JOIN `jobs` ON jobs.batch_id = `job_parents`.batch_id AND jobs.job_id = `job_parents`.parent_id
          WHERE `job_parents`.batch_id = in_batch_id
          GROUP BY `job_parents`.batch_id, `job_parents`.job_id
          FOR UPDATE
        ) AS t
          ON jobs.batch_id = t.batch_id AND
             jobs.job_id = t.job_id
        SET jobs.state = IF(t.n_pending_parents = 0, 'Ready', 'Pending'),
            jobs.n_pending_parents = t.n_pending_parents,
            jobs.cancelled = IF(t.n_pending_parents > 0 OR t.n_succeeded = t.n_parents, jobs.cancelled, 1)
        WHERE jobs.batch_id = in_batch_id AND
              jobs.committed = 0;
    END IF;

    DELETE FROM batches_inst_coll_staging WHERE batch_id = in_batch_id;

    COMMIT;
    SELECT 0 as rc;
  END IF;
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

    UPDATE batches SET n_completed = n_completed + 1, time_last_updated = new_timestamp WHERE id = in_batch_id;

    UPDATE batches
      SET time_completed = IF(closed, new_timestamp, NULL),
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

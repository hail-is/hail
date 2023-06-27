CREATE TABLE IF NOT EXISTS `jobs_telemetry` (
  `batch_id` BIGINT NOT NULL,
  `job_id` INT NOT NULL,
  `time_ready` BIGINT DEFAULT NULL,
  PRIMARY KEY (`batch_id`, `job_id`),
  FOREIGN KEY (`batch_id`) REFERENCES batches(id) ON DELETE CASCADE,
) ENGINE = InnoDB;

DELIMITER $$

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
    WHERE batch_id = in_batch_id AND update_id = in_update_id
    FOR UPDATE;

    IF staging_n_jobs = expected_n_jobs THEN
      UPDATE batch_updates
      SET committed = 1, time_committed = in_timestamp
      WHERE batch_id = in_batch_id AND update_id = in_update_id;

      UPDATE batches SET
        `state` = 'running',
        time_completed = NULL,
        n_jobs = n_jobs + expected_n_jobs
      WHERE id = in_batch_id;

      INSERT INTO user_inst_coll_resources (user, inst_coll, token, n_ready_jobs, ready_cores_mcpu)
      SELECT user, inst_coll, 0, @n_ready_jobs := COALESCE(SUM(n_ready_jobs), 0), @ready_cores_mcpu := COALESCE(SUM(ready_cores_mcpu), 0)
      FROM batches_inst_coll_staging
      JOIN batches ON batches.id = batches_inst_coll_staging.batch_id
      WHERE batch_id = in_batch_id AND update_id = in_update_id
      GROUP BY `user`, inst_coll
      ON DUPLICATE KEY UPDATE
        n_ready_jobs = n_ready_jobs + @n_ready_jobs,
        ready_cores_mcpu = ready_cores_mcpu + @ready_cores_mcpu;

      DELETE FROM batches_inst_coll_staging WHERE batch_id = in_batch_id AND update_id = in_update_id;

      IF in_update_id != 1 THEN
        SELECT start_job_id INTO cur_update_start_job_id FROM batch_updates WHERE batch_id = in_batch_id AND update_id = in_update_id;

        UPDATE jobs, jobs_telemetry
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
  DECLARE total_jobs_in_batch INT;
  DECLARE expected_attempt_id VARCHAR(40);

  START TRANSACTION;

  SELECT n_jobs INTO total_jobs_in_batch FROM batches WHERE id = in_batch_id;

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

    UPDATE batches_n_jobs_in_complete_states
      SET n_completed = (@new_n_completed := n_completed + 1),
          n_cancelled = n_cancelled + (new_state = 'Cancelled'),
          n_failed    = n_failed + (new_state = 'Error' OR new_state = 'Failed'),
          n_succeeded = n_succeeded + (new_state != 'Cancelled' AND new_state != 'Error' AND new_state != 'Failed')
      WHERE id = in_batch_id;

    # Grabbing an exclusive lock on batches here could deadlock,
    # but this IF should only execute for the last job
    IF @new_n_completed = total_jobs_in_batch THEN
      UPDATE batches
      SET time_completed = new_timestamp,
          `state` = 'complete'
      WHERE id = in_batch_id;
    END IF;

    UPDATE jobs, jobs_telemetry
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

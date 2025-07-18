CREATE TABLE IF NOT EXISTS `batches_n_jobs_in_complete_states` (
  `id` BIGINT NOT NULL,
  `n_completed` INT NOT NULL DEFAULT 0,
  `n_succeeded` INT NOT NULL DEFAULT 0,
  `n_failed` INT NOT NULL DEFAULT 0,
  `n_cancelled` INT NOT NULL DEFAULT 0,
  PRIMARY KEY (`id`),
  FOREIGN KEY (`id`) REFERENCES batches(id) ON DELETE CASCADE
) ENGINE = InnoDB;


INSERT INTO batches_n_jobs_in_complete_states
SELECT id, n_completed, n_succeeded, n_failed, n_cancelled
FROM batches;

ALTER TABLE batches
DROP COLUMN n_completed,
DROP COLUMN n_succeeded,
DROP COLUMN n_failed,
DROP COLUMN n_cancelled;

DELIMITER $$

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
  SET start_time = new_start_time, end_time = new_end_time, reason = new_reason
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

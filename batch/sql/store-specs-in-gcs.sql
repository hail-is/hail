ALTER TABLE batches ADD COLUMN format_version INT NOT NULL DEFAULT 1;
ALTER TABLE batches MODIFY COLUMN format_version INT NOT NULL;

CREATE TABLE IF NOT EXISTS `batch_bunches` (
  `batch_id` BIGINT NOT NULL,
  `start_job_id` INT NOT NULL,
  `token` VARCHAR(100) NOT NULL,
  PRIMARY KEY (`batch_id`, `start_job_id`),
  FOREIGN KEY (`batch_id`, `start_job_id`) REFERENCES jobs(batch_id, job_id) ON DELETE CASCADE
) ENGINE = InnoDB;

ALTER TABLE instances ADD COLUMN version INT NOT NULL DEFAULT 1;
ALTER TABLE instances MODIFY COLUMN version INT NOT NULL;

DELIMITER $$

DROP PROCEDURE IF EXISTS mark_job_complete;
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

  SELECT state INTO cur_instance_state FROM instances WHERE name = in_instance_name FOR UPDATE;
  IF cur_instance_state = 'active' AND cur_end_time IS NULL THEN
    UPDATE instances
    SET free_cores_mcpu = free_cores_mcpu + cur_cores_mcpu
    WHERE name = in_instance_name;

    SET delta_cores_mcpu = delta_cores_mcpu + cur_cores_mcpu;
  END IF;

  SELECT attempt_id INTO expected_attempt_id FROM jobs
  WHERE batch_id = in_batch_id AND job_id = in_job_id
  FOR UPDATE;

  IF expected_attempt_id != in_attempt_id THEN
    COMMIT;
    SELECT 2 as rc,
      expected_attempt_id,
      delta_cores_mcpu,
      'input attempt id does not match expected attempt id' as message;
  ELSEIF cur_job_state = 'Ready' OR cur_job_state = 'Running' THEN
    UPDATE jobs
    SET state = new_state, status = new_status
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
      'job state not Ready, Running or complete' as message;
  END IF;
END $$

DROP TRIGGER IF EXISTS attempts_before_update;
CREATE TRIGGER attempts_before_update BEFORE UPDATE ON attempts
FOR EACH ROW
BEGIN
  IF OLD.start_time IS NOT NULL AND (NEW.start_time IS NULL OR NEW.start_time < OLD.start_time) THEN
    SET NEW.start_time = OLD.start_time;
  END IF;

  IF OLD.reason IS NOT NULL AND (OLD.end_time IS NULL OR NEW.end_time IS NULL OR NEW.end_time >= OLD.end_time) THEN
    SET NEW.end_time = OLD.end_time;
    SET NEW.reason = OLD.reason;
  END IF;
END $$

DELIMITER ;

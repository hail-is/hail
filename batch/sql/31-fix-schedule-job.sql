DELIMITER $$

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

  SELECT jobs.state, cores_mcpu, attempt_id,
    (jobs.cancelled OR batches.cancelled) AND NOT always_run
  INTO cur_job_state, cur_cores_mcpu, cur_attempt_id, cur_job_cancel
  FROM jobs
  INNER JOIN batches ON batches.id = jobs.batch_id
  WHERE batch_id = in_batch_id AND job_id = in_job_id
  FOR UPDATE;

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

DELIMITER ;

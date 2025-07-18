DELIMITER $$

DROP TRIGGER IF EXISTS attempt_resources_after_insert $$
CREATE TRIGGER attempt_resources_after_insert AFTER INSERT ON attempt_resources
FOR EACH ROW
BEGIN
  DECLARE cur_start_time BIGINT;
  DECLARE cur_end_time BIGINT;
  DECLARE cur_billing_project VARCHAR(100);
  DECLARE msec_diff BIGINT;
  DECLARE cur_n_tokens INT;
  DECLARE rand_token INT;

  SELECT n_tokens INTO cur_n_tokens FROM globals LOCK IN SHARE MODE;
  SET rand_token = FLOOR(RAND() * cur_n_tokens);

  SELECT billing_project INTO cur_billing_project FROM batches WHERE id = NEW.batch_id;

  SELECT start_time, end_time INTO cur_start_time, cur_end_time
  FROM attempts
  WHERE batch_id = NEW.batch_id AND job_id = NEW.job_id AND attempt_id = NEW.attempt_id
  LOCK IN SHARE MODE;

  SET msec_diff = GREATEST(COALESCE(cur_end_time - cur_start_time, 0), 0);

  INSERT INTO aggregated_billing_project_resources (billing_project, resource, token, `usage`)
  VALUES (cur_billing_project, NEW.resource, rand_token, NEW.quantity * msec_diff)
  ON DUPLICATE KEY UPDATE
    `usage` = `usage` + NEW.quantity * msec_diff;

  INSERT INTO aggregated_batch_resources (batch_id, resource, token, `usage`)
  VALUES (NEW.batch_id, NEW.resource, rand_token, NEW.quantity * msec_diff)
  ON DUPLICATE KEY UPDATE
    `usage` = `usage` + NEW.quantity * msec_diff;

  INSERT INTO aggregated_job_resources (batch_id, job_id, resource, `usage`)
  VALUES (NEW.batch_id, NEW.job_id, NEW.resource, NEW.quantity * msec_diff)
  ON DUPLICATE KEY UPDATE
    `usage` = `usage` + NEW.quantity * msec_diff;
END $$

DROP PROCEDURE IF EXISTS add_attempt $$
CREATE PROCEDURE add_attempt(
  IN in_batch_id BIGINT,
  IN in_job_id INT,
  IN in_attempt_id VARCHAR(40),
  IN in_instance_name VARCHAR(100),
  IN in_cores_mcpu INT,
  OUT delta_cores_mcpu INT
)
BEGIN
  DECLARE cur_instance_state VARCHAR(40);
  DECLARE dummy_lock INT;

  SET delta_cores_mcpu = IFNULL(delta_cores_mcpu, 0);

  IF in_attempt_id IS NOT NULL THEN
    SELECT 1 INTO dummy_lock FROM instances_free_cores_mcpu
    WHERE instances_free_cores_mcpu.name = in_instance_name
    FOR UPDATE;

    INSERT INTO attempts (batch_id, job_id, attempt_id, instance_name)
    VALUES (in_batch_id, in_job_id, in_attempt_id, in_instance_name)
    ON DUPLICATE KEY UPDATE batch_id = batch_id;

    IF ROW_COUNT() = 1 THEN
      SELECT state INTO cur_instance_state
      FROM instances
      WHERE name = in_instance_name
      LOCK IN SHARE MODE;

      IF cur_instance_state = 'pending' OR cur_instance_state = 'active' THEN
        UPDATE instances_free_cores_mcpu
        SET free_cores_mcpu = free_cores_mcpu - in_cores_mcpu
        WHERE instances_free_cores_mcpu.name = in_instance_name;
      END IF;

      SET delta_cores_mcpu = -1 * in_cores_mcpu;
    END IF;
  END IF;
END $$

DELIMITER ;

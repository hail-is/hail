DELIMITER $$

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
  SET delta_cores_mcpu = IFNULL(delta_cores_mcpu, 0);

  IF in_attempt_id IS NOT NULL THEN
    INSERT INTO attempts (batch_id, job_id, attempt_id, instance_name)
    VALUES (in_batch_id, in_job_id, in_attempt_id, in_instance_name)
    ON DUPLICATE KEY UPDATE batch_id = batch_id;

    IF ROW_COUNT() != 0 THEN
      UPDATE instances, instances_free_cores_mcpu
      SET free_cores_mcpu = free_cores_mcpu - in_cores_mcpu
      WHERE instances.name = in_instance_name
        AND instances.name = instances_free_cores_mcpu.name
        AND (instances.state = 'pending' OR instances.state = 'active');

      SET delta_cores_mcpu = -1 * in_cores_mcpu;
    END IF;
  END IF;
END $$

DELIMITER ;

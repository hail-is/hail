ALTER TABLE batches ADD COLUMN time_closed BIGINT;

DELIMITER $$

DROP PROCEDURE IF EXISTS close_batch;
CREATE PROCEDURE close_batch(
  IN in_batch_id BIGINT,
  IN in_timestamp BIGINT
)
BEGIN
  DECLARE cur_batch_state VARCHAR(40);
  DECLARE expected_n_jobs INT;
  DECLARE staging_n_jobs INT;
  DECLARE staging_n_ready_jobs INT;
  DECLARE staging_ready_cores_mcpu BIGINT;
  DECLARE cur_user VARCHAR(100);

  START TRANSACTION;

  SELECT `state`, n_jobs INTO cur_batch_state, expected_n_jobs FROM batches
  WHERE id = in_batch_id AND NOT deleted
  FOR UPDATE;

  IF cur_batch_state != 'open' THEN
    COMMIT;
    SELECT 0 as rc;
  ELSE
    SELECT COALESCE(SUM(n_jobs), 0), COALESCE(SUM(n_ready_jobs), 0), COALESCE(SUM(ready_cores_mcpu), 0)
    INTO staging_n_jobs, staging_n_ready_jobs, staging_ready_cores_mcpu
    FROM batches_staging
    WHERE batch_id = in_batch_id
    FOR UPDATE;

    SELECT user INTO cur_user FROM batches WHERE id = in_batch_id;

    IF staging_n_jobs = expected_n_jobs THEN
      IF expected_n_jobs = 0 THEN
        UPDATE batches SET `state` = 'complete', time_completed = in_timestamp, time_closed = in_timestamp
          WHERE id = in_batch_id;
      ELSE
        UPDATE batches SET `state` = 'running', time_closed = in_timestamp
          WHERE id = in_batch_id;
      END IF;

      INSERT INTO ready_cores (token, ready_cores_mcpu)
      VALUES (0, staging_ready_cores_mcpu)
      ON DUPLICATE KEY UPDATE ready_cores_mcpu = ready_cores_mcpu + staging_ready_cores_mcpu;

      INSERT INTO user_resources (user, token, n_ready_jobs, ready_cores_mcpu)
      VALUES (cur_user, 0, staging_n_ready_jobs, staging_ready_cores_mcpu)
      ON DUPLICATE KEY UPDATE
        n_ready_jobs = n_ready_jobs + staging_n_ready_jobs,
        ready_cores_mcpu = ready_cores_mcpu + staging_ready_cores_mcpu;

      DELETE FROM batches_staging WHERE batch_id = in_batch_id;

      COMMIT;
      SELECT 0 as rc;
    ELSE
      ROLLBACK;
      SELECT 2 as rc, expected_n_jobs, staging_n_jobs as actual_n_jobs, 'wrong number of jobs' as message;
    END IF;
  END IF;
END $$

DELIMITER ;

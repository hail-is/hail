SET @n_tokens = 200;

ALTER TABLE globals ADD COLUMN n_tokens INT NOT NULL;
UPDATE globals SET n_tokens = @n_tokens;

CREATE TABLE IF NOT EXISTS `batches_staging` (
  `batch_id` BIGINT NOT NULL,
  `token` INT NOT NULL,
  `n_jobs` INT NOT NULL DEFAULT 0,
  `n_ready_jobs` INT NOT NULL DEFAULT 0,
  `ready_cores_mcpu` BIGINT NOT NULL DEFAULT 0,
  PRIMARY KEY (`batch_id`, `token`),
  FOREIGN KEY (`batch_id`) REFERENCES batches(`id`) ON DELETE CASCADE
) ENGINE = InnoDB;

ALTER TABLE user_resources ADD COLUMN token INT NOT NULL DEFAULT 0;
ALTER TABLE user_resources DROP PRIMARY KEY, ADD PRIMARY KEY(`user`, `token`);

ALTER TABLE ready_cores ADD COLUMN token INT NOT NULL DEFAULT 0;
ALTER TABLE ready_cores ADD PRIMARY KEY(`token`);

CREATE TEMPORARY TABLE tmp_resources AS (
  SELECT batch_id, user, closed,
    0 as token,
    COUNT(*) AS n_jobs,
    COALESCE(SUM(state = 'Ready'), 0) AS n_ready_jobs,
    COALESCE(SUM(state = 'Running'), 0) AS n_running_jobs,
    COALESCE(SUM(IF(state = 'Ready', cores_mcpu, 0)), 0) AS ready_cores_mcpu,
    COALESCE(SUM(IF(state = 'Running', cores_mcpu, 0)), 0) AS running_cores_mcpu
  FROM jobs
  INNER JOIN batches ON batches.id = jobs.batch_id
  GROUP BY batch_id, user, closed
);

SELECT COALESCE(SUM(ready_cores_mcpu), 0) INTO @closed_ready_cores_mcpu
FROM tmp_resources
WHERE closed;

UPDATE ready_cores SET ready_cores_mcpu = @closed_ready_cores_mcpu;

UPDATE user_resources
RIGHT JOIN (
  SELECT user, token,
    COALESCE(SUM(n_ready_jobs), 0) as n_ready_jobs,
    COALESCE(SUM(n_running_jobs), 0) as n_running_jobs,
    COALESCE(SUM(ready_cores_mcpu), 0) as ready_cores_mcpu,
    COALESCE(SUM(running_cores_mcpu), 0) as running_cores_mcpu
  FROM tmp_resources
  WHERE closed
  GROUP BY user, token) AS t
ON user_resources.user = t.user AND user_resources.token = t.token
SET
  user_resources.n_ready_jobs = t.n_ready_jobs,
  user_resources.n_running_jobs = t.n_running_jobs,
  user_resources.ready_cores_mcpu = t.ready_cores_mcpu,
  user_resources.running_cores_mcpu = t.running_cores_mcpu;

INSERT INTO batches_staging (batch_id, token, n_jobs, n_ready_jobs, ready_cores_mcpu)
  SELECT tmp_resources.batch_id,
    tmp_resources.token,
    tmp_resources.n_jobs,
    tmp_resources.n_ready_jobs,
    tmp_resources.ready_cores_mcpu
  FROM tmp_resources
  WHERE NOT closed;

DELIMITER $$

DROP TRIGGER IF EXISTS jobs_after_update;
CREATE TRIGGER jobs_after_update AFTER UPDATE ON jobs
FOR EACH ROW
BEGIN
  DECLARE in_user VARCHAR(100);
  DECLARE cur_n_tokens INT;
  DECLARE rand_token INT;

  SELECT user INTO in_user from batches
  WHERE id = NEW.batch_id;

  SELECT n_tokens INTO cur_n_tokens FROM globals LOCK IN SHARE MODE;
  SET rand_token = FLOOR(RAND() * cur_n_tokens);

  IF OLD.state = 'Ready' THEN
    INSERT INTO user_resources (user, token, n_ready_jobs, ready_cores_mcpu) VALUES (in_user, rand_token, -1, -1 * OLD.cores_mcpu)
    ON DUPLICATE KEY UPDATE
      n_ready_jobs = n_ready_jobs - 1,
      ready_cores_mcpu = ready_cores_mcpu - OLD.cores_mcpu;

    INSERT INTO ready_cores (token, ready_cores_mcpu) VALUES (rand_token, -1 * OLD.cores_mcpu)
    ON DUPLICATE KEY UPDATE
      ready_cores_mcpu = ready_cores_mcpu - OLD.cores_mcpu;
  END IF;

  IF NEW.state = 'Ready' THEN
    INSERT INTO user_resources (user, token, n_ready_jobs, ready_cores_mcpu) VALUES (in_user, rand_token, 1, NEW.cores_mcpu)
    ON DUPLICATE KEY UPDATE
      n_ready_jobs = n_ready_jobs + 1,
      ready_cores_mcpu = ready_cores_mcpu + NEW.cores_mcpu;

    INSERT INTO ready_cores (token, ready_cores_mcpu) VALUES (rand_token, NEW.cores_mcpu)
    ON DUPLICATE KEY UPDATE
      ready_cores_mcpu = ready_cores_mcpu + NEW.cores_mcpu;
  END IF;

  IF OLD.state = 'Running' THEN
    INSERT INTO user_resources (user, token, n_running_jobs, running_cores_mcpu) VALUES (in_user, rand_token, -1, -1 * OLD.cores_mcpu)
    ON DUPLICATE KEY UPDATE
      n_running_jobs = n_running_jobs - 1,
      running_cores_mcpu = running_cores_mcpu - OLD.cores_mcpu;
  END IF;

  IF NEW.state = 'Running' THEN
    INSERT INTO user_resources (user, token, n_running_jobs, running_cores_mcpu) VALUES (in_user, rand_token, 1, NEW.cores_mcpu)
    ON DUPLICATE KEY UPDATE
      n_running_jobs = n_running_jobs + 1,
      running_cores_mcpu = running_cores_mcpu + NEW.cores_mcpu;
  END IF;
END $$

DROP PROCEDURE IF EXISTS close_batch;
CREATE PROCEDURE close_batch(
  IN in_batch_id BIGINT,
  IN in_timestamp BIGINT
)
BEGIN
  DECLARE cur_batch_closed BOOLEAN;
  DECLARE expected_n_jobs INT;
  DECLARE staging_n_jobs INT;
  DECLARE staging_n_ready_jobs INT;
  DECLARE staging_ready_cores_mcpu BIGINT;
  DECLARE cur_user VARCHAR(100);

  START TRANSACTION;

  SELECT n_jobs, closed INTO expected_n_jobs, cur_batch_closed FROM batches
  WHERE id = in_batch_id AND NOT deleted
  FOR UPDATE;

  IF cur_batch_closed THEN
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
      UPDATE batches SET closed = 1 WHERE id = in_batch_id;
      UPDATE batches SET time_completed = in_timestamp
        WHERE id = in_batch_id AND n_completed = batches.n_jobs;

      INSERT INTO ready_cores (token, ready_cores_mcpu) VALUES (0, staging_ready_cores_mcpu)
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

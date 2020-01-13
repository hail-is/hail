ALTER TABLE user_resources ADD COLUMN token INT NOT NULL DEFAULT 0;
ALTER TABLE user_resources DROP PRIMARY KEY, ADD PRIMARY KEY(`user`, `token`);

ALTER TABLE ready_cores ADD COLUMN token INT NOT NULL DEFAULT 0;
ALTER TABLE ready_cores ADD PRIMARY KEY(`token`);

CREATE TABLE IF NOT EXISTS `batches_staging` (
  `batch_id` BIGINT NOT NULL,
  `token` INT NOT NULL,
  `n_jobs` INT NOT NULL DEFAULT 0,
  `n_ready_jobs` INT NOT NULL DEFAULT 0,
  `ready_cores_mcpu` BIGINT NOT NULL DEFAULT 0,
  PRIMARY KEY (`batch_id`, `token`),
  FOREIGN KEY (`batch_id`) REFERENCES batches(`id`) ON DELETE CASCADE
) ENGINE = InnoDB;

DELIMITER $$

DROP PROCEDURE IF EXISTS insert_batches_staging_tokens;
CREATE PROCEDURE insert_batches_staging_tokens(
  IN in_batch_id BIGINT
)
BEGIN
  DECLARE i int DEFAULT 0;
  DECLARE row_exists BOOLEAN;

  WHILE i < 32 DO
    SET row_exists = EXISTS(SELECT * FROM batches_staging WHERE batch_id = in_batch_id AND token = i FOR UPDATE);
    IF NOT row_exists THEN
      INSERT INTO batches_staging (batch_id, token) VALUES (in_batch_id, i);
    END IF;
    SET i = i + 1;
  END WHILE;
END $$

DROP PROCEDURE IF EXISTS insert_ready_cores_tokens;
CREATE PROCEDURE insert_ready_cores_tokens()
BEGIN
  DECLARE i int DEFAULT 0;
  DECLARE row_exists BOOLEAN;

  WHILE i < 32 DO
    SET row_exists = EXISTS(SELECT * FROM ready_cores WHERE token = i FOR UPDATE);
    IF NOT row_exists THEN
      INSERT INTO ready_cores (token, ready_cores_mcpu) VALUES (i, 0);
    END IF;
    SET i = i + 1;
  END WHILE;
END $$

DROP PROCEDURE IF EXISTS insert_user_resources_tokens;
CREATE PROCEDURE insert_user_resources_tokens(
  IN in_user VARCHAR(100)
)
BEGIN
  DECLARE i int DEFAULT 0;
  DECLARE row_exists BOOLEAN;

  WHILE i < 32 DO
    SET row_exists = EXISTS(SELECT * FROM user_resources WHERE user = in_user AND token = i FOR UPDATE);
    IF NOT row_exists THEN
      INSERT INTO user_resources (user, token) VALUES (in_user, i);
    END IF;
    SET i = i + 1;
  END WHILE;
END $$

DROP TRIGGER IF EXISTS jobs_after_update;
CREATE TRIGGER jobs_after_update AFTER UPDATE ON jobs
FOR EACH ROW
BEGIN
  DECLARE in_user VARCHAR(100);
  DECLARE rand_token INT;

  SELECT user INTO in_user from batches
  WHERE id = NEW.batch_id;

  SET rand_token = FLOOR(RAND() * 32);

  IF OLD.state = 'Ready' THEN
    UPDATE user_resources
      SET n_ready_jobs = n_ready_jobs - 1, ready_cores_mcpu = ready_cores_mcpu - OLD.cores_mcpu
      WHERE user = in_user AND token = rand_token;

    UPDATE ready_cores
      SET ready_cores_mcpu = ready_cores_mcpu - OLD.cores_mcpu
      WHERE token = rand_token;
  END IF;

  IF NEW.state = 'Ready' THEN
    UPDATE user_resources
      SET n_ready_jobs = n_ready_jobs + 1, ready_cores_mcpu = ready_cores_mcpu + NEW.cores_mcpu
      WHERE user = in_user AND token = rand_token;

    UPDATE ready_cores
      SET ready_cores_mcpu = ready_cores_mcpu + NEW.cores_mcpu
      WHERE token = rand_token;
  END IF;

  IF OLD.state = 'Running' THEN
    UPDATE user_resources
    SET n_running_jobs = n_running_jobs - 1, running_cores_mcpu = running_cores_mcpu - OLD.cores_mcpu
    WHERE user = in_user AND token = rand_token;
  END IF;

  IF NEW.state = 'Running' THEN
    UPDATE user_resources
    SET n_running_jobs = n_running_jobs + 1, running_cores_mcpu = running_cores_mcpu + NEW.cores_mcpu
    WHERE user = in_user AND token = rand_token;
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
  DECLARE staging_ready_cores_mcpu INT;
  DECLARE cur_user VARCHAR(100);

  START TRANSACTION;

  SELECT n_jobs, closed INTO expected_n_jobs, cur_batch_closed FROM batches
  WHERE id = in_batch_id AND NOT deleted;

  IF cur_batch_closed = 1 THEN
    COMMIT;
    SELECT 0 as rc;
  ELSEIF cur_batch_closed = 0 THEN
    SELECT SUM(n_jobs), SUM(n_ready_jobs), SUM(ready_cores_mcpu)
    INTO staging_n_jobs, staging_n_ready_jobs, staging_ready_cores_mcpu
    FROM batches_staging
    WHERE batch_id = in_batch_id
    FOR UPDATE;

    SELECT user INTO cur_user FROM batches WHERE id = in_batch_id;

    IF staging_n_jobs = expected_n_jobs THEN
      UPDATE batches SET closed = 1 WHERE id = in_batch_id;
      UPDATE batches SET time_completed = in_timestamp
        WHERE id = in_batch_id AND n_completed = batches.n_jobs;

      UPDATE ready_cores
      SET ready_cores_mcpu = ready_cores_mcpu + staging_ready_cores_mcpu
      WHERE token = 0;

      UPDATE user_resources
      SET n_ready_jobs = n_ready_jobs + staging_n_ready_jobs,
         ready_cores_mcpu = ready_cores_mcpu + staging_ready_cores_mcpu
      WHERE user = cur_user AND token = 0;

      DELETE FROM batches_staging WHERE batch_id = in_batch_id;

      COMMIT;
      SELECT 0 as rc;
    ELSE
      ROLLBACK;
      SELECT 2 as rc, expected_n_jobs, staging_n_jobs as actual_n_jobs, 'wrong number of jobs' as message;
    END IF;
  ELSE
    ROLLBACK;
    SELECT 1 as rc, cur_batch_closed, 'batch closed is not 0 or 1' as message;
  END IF;
END $$

DELIMITER ;
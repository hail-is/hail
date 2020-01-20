
CREATE TABLE `batch_ready_cancellable` (
  `batch_id` BIGINT NOT NULL,
  `token` INT NOT NULL,
  # neither run_always nor cancelled
  `n_ready_cancellable_jobs` INT NOT NULL DEFAULT 0,
  `ready_cancellable_cores_mcpu` BIGINT NOT NULL DEFAULT 0,
  PRIMARY KEY (`batch_id`, `token`),
  FOREIGN KEY (`batch_id`) REFERENCES batches(id) ON DELETE CASCADE
) ENGINE = InnoDB;

DELIMITER $$

CREATE PROCEDURE recompute_incremental(
) BEGIN

  START TRANSACTION;

  DELETE FROM batch_staging;
  DELETE FROM batch_ready_cancellable;
  DELETE FROM ready_cores;
  DELETE FROM user_resources;

  INSERT INTO batch_staging (batch_id, token, n_jobs, n_ready_jobs, ready_cores_mcpu)
  SELECT t.batch_id, 0, t.n_jobs, t.n_ready_jobs, t.ready_cores_mcpu
  FROM (SELECT batch_id,
      COALESCE(SUM(1), 0) as n_jobs,
      COALESCE(SUM(state = 'Ready'), 0) as n_ready_jobs,
      COALESCE(SUM(IF(state = 'Ready', cores_mcpu, 0)), 0) as ready_cores_mcpu
    FROM (SELECT
        batches.id as batch_id,
	jobs.state,
	jobs.cores_mcpu
      FROM jobs
      INNER JOIN batches ON batches.id = jobs.batch_id
      WHERE NOT batches.closed) AS s
    GROUP BY batch_id) as t;

  INSERT INTO batch_ready_cancellable (batch_id, token, n_ready_cancellable_jobs, ready_cancellable_cores_mcpu)
  SELECT t.batch_id, 0, t.n_ready_cancellable_jobs, t.ready_cancellable_cores_mcpu
  FROM (SELECT batch_id,
      COALESCE(SUM(cancellable), 0) as n_ready_cancellable_jobs,
      COALESCE(SUM(IF(cancellable, cores_mcpu, 0)), 0) as ready_cancellable_cores_mcpu
    FROM (SELECT
        batches.id as batch_id,
	jobs.cores_mcpu,
        NOT (jobs.always_run OR jobs.cancelled OR batches.cancelled) AS cancellable
      FROM jobs
      INNER JOIN batches ON batches.id = jobs.batch_id
      WHERE batches.closed
        AND jobs.state = 'Ready') AS s
    GROUP BY batch_id) as t;

  INSERT INTO ready_cores (token, ready_cores_mcpu)
  SELECT 0, t.ready_cores_mcpu
  FROM (SELECT COALESCE(SUM(cores_mcpu), 0) as ready_cores_mcpu
    FROM jobs
    INNER JOIN batches ON batches.id = jobs.batch_id
    WHERE batches.closed
	    AND jobs.state = 'Ready'
	    # runnable
	    AND (jobs.always_run OR NOT (jobs.cancelled OR batches.cancelled))) as t;

  INSERT INTO user_resources (token, user, n_ready_jobs, ready_cores_mcpu, n_running_jobs, running_cores_mcpu)
  SELECT 0, t.user, t.n_ready_jobs, t.ready_cores_mcpu,t.n_running_jobs, t.running_cores_mcpu
  FROM (SELECT user,
      COALESCE(SUM(state = 'Running'), 0) as n_running_jobs,
      COALESCE(SUM(IF(state = 'Running', cores_mcpu, 0)), 0) as running_cores_mcpu,
      COALESCE(SUM(state = 'Ready' AND runnable), 0) as n_ready_jobs,
      COALESCE(SUM(IF(state = 'Ready' AND runnable, cores_mcpu, 0)), 0) as ready_cores_mcpu
    FROM (SELECT
        jobs.state,
	jobs.cores_mcpu,
        (jobs.always_run OR NOT (jobs.cancelled OR batches.cancelled)) AS runnable,
        batches.user
      FROM jobs
      INNER JOIN batches ON batches.id = jobs.batch_id
      WHERE batches.closed) AS s
    GROUP BY user) as t;

  COMMIT;

END $$

CREATE PROCEDURE cancel_batch(
  IN in_batch_id VARCHAR(100)
)
BEGIN
  DECLARE cur_user VARCHAR(100);
  DECLARE cur_closed BOOLEAN;
  DECLARE cur_cancelled BOOLEAN;
  DECLARE cur_n_ready_cancelled_jobs INT;
  DECLARE cur_ready_cancelled_cores_mcpu BIGINT;

  START TRANSACTION;

  SELECT user, closed, cancelled INTO cur_user, cur_closed, cur_cancelled FROM batches
  WHERE id = in_batch_id
  FOR UPDATE;

  IF cur_closed AND NOT cur_cancelled THEN
    SELECT COALESCE(SUM(n_ready_cancellable_jobs), 0), COALESCE(SUM(ready_cancellable_cores_mcpu), 0)
    INTO cur_n_ready_cancelled_jobs, cur_ready_cancelled_cores_mcpu
    FROM batch_ready_cancellable
    WHERE batch_id = in_batch_id;

    INSERT INTO user_resources (user, token, n_ready_jobs, ready_cores_mcpu)
    VALUES (cur_user, 0, -cur_n_ready_cancelled_jobs, -cur_ready_cancelled_cores_mcpu)
    ON DUPLICATE KEY UPDATE
      n_ready_jobs = n_ready_jobs - cur_n_ready_cancelled_jobs,
      ready_cores_mcpu = ready_cores_mcpu - cur_ready_cancelled_cores_mcpu;

    INSERT INTO ready_cores (token, ready_cores_mcpu)
    VALUES (0, -cur_ready_cancelled_cores_mcpu)
    ON DUPLICATE KEY UPDATE
      ready_cores_mcpu = ready_cores_mcpu - cur_ready_cancelled_cores_mcpu;

    # there are no cancellable left, they have been cancelled
    DELETE FROM batch_ready_cancellable WHERE batch_id = in_batch_id;

    UPDATE batches SET cancelled = 1 WHERE id = in_batch_id;
  END IF;

  COMMIT;
END $$

DROP TRIGGER IF EXISTS jobs_after_update;
CREATE TRIGGER jobs_after_update AFTER UPDATE ON jobs
FOR EACH ROW
BEGIN
  DECLARE cur_user VARCHAR(100);
  DECLARE cur_batch_cancelled BOOLEAN;
  DECLARE cur_n_tokens INT;
  DECLARE rand_token INT;

  SELECT user, cancelled INTO cur_user, cur_batch_cancelled FROM batches
  WHERE id = NEW.batch_id;

  SELECT n_tokens INTO cur_n_tokens FROM globals LOCK IN SHARE MODE;
  SET rand_token = FLOOR(RAND() * cur_n_tokens);

  IF OLD.state = 'Ready' THEN
    # cancellable (and not cancelled)
    IF NOT (OLD.always_run OR OLD.cancelled OR cur_batch_cancelled) THEN
      INSERT INTO batch_ready_cancellable (batch_id, token, n_ready_cancellable_jobs, ready_cancellable_cores_mcpu)
      VALUES (OLD.batch_id, rand_token, -1, -OLD.cores_mcpu)
      ON DUPLICATE KEY UPDATE
	n_ready_cancellable_jobs = n_ready_cancellable_jobs - 1,
	ready_cancellable_cores_mcpu = ready_cancellable_cores_mcpu - OLD.cores_mcpu;
    END IF;

    # runnable
    IF OLD.always_run OR NOT (OLD.cancelled OR cur_batch_cancelled) THEN
      INSERT INTO user_resources (user, token, n_ready_jobs, ready_cores_mcpu) VALUES (cur_user, rand_token, -1, -OLD.cores_mcpu)
      ON DUPLICATE KEY UPDATE
	n_ready_jobs = n_ready_jobs - 1,
	ready_cores_mcpu = ready_cores_mcpu - OLD.cores_mcpu;

      INSERT INTO ready_cores (token, ready_cores_mcpu) VALUES (rand_token, -OLD.cores_mcpu)
      ON DUPLICATE KEY UPDATE
	ready_cores_mcpu = ready_cores_mcpu - OLD.cores_mcpu;
    ENDIF;
  ELSEIF OLD.state = 'Running' THEN
    INSERT INTO user_resources (user, token, n_running_jobs, running_cores_mcpu) VALUES (cur_user, rand_token, -1, -OLD.cores_mcpu)
    ON DUPLICATE KEY UPDATE
      n_running_jobs = n_running_jobs - 1,
      running_cores_mcpu = running_cores_mcpu - OLD.cores_mcpu;
  END IF;

  IF NEW.state = 'Ready' THEN
    # cancellable (and not cancelled)
    IF NOT (NEW.always_run OR NEW.cancelled OR cur_batch_cancelled) THEN
      INSERT INTO batch_ready_cancellable (batch_id, token, n_ready_cancellable_jobs, ready_cancellable_cores_mcpu)
      VALUES (NEW.batch_id, rand_token, 1, NEW.cores_mcpu)
      ON DUPLICATE KEY UPDATE
	n_ready_cancellable_jobs = n_ready_cancellable_jobs + 1,
	ready_cancellable_cores_mcpu = ready_cancellable_cores_mcpu + NEW.cores_mcpu;
    END IF;

    # runnable
    IF NEW.always_run OR NOT (NEW.cancelled OR cur_batch_cancelled) THEN
      INSERT INTO user_resources (user, token, n_ready_jobs, ready_cores_mcpu) VALUES (cur_user, rand_token, 1, NEW.cores_mcpu)
      ON DUPLICATE KEY UPDATE
	n_ready_jobs = n_ready_jobs + 1,
	ready_cores_mcpu = ready_cores_mcpu + NEW.cores_mcpu;

      INSERT INTO ready_cores (token, ready_cores_mcpu) VALUES (rand_token, NEW.cores_mcpu)
      ON DUPLICATE KEY UPDATE
	ready_cores_mcpu = ready_cores_mcpu + NEW.cores_mcpu;
    END IF;
  ELSEIF NEW.state = 'Running' THEN
    INSERT INTO user_resources (user, token, n_running_jobs, running_cores_mcpu) VALUES (cur_user, rand_token, 1, NEW.cores_mcpu)
    ON DUPLICATE KEY UPDATE
      n_running_jobs = n_running_jobs + 1,
      running_cores_mcpu = running_cores_mcpu + NEW.cores_mcpu;
  END IF;
END $$

DELIMITER ;

CALL recompute_incremental();

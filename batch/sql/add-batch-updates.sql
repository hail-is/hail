START TRANSACTION;

# Foreign key constraint for batch_id needs to be added in a subsequent PR in order for the migration to work
CREATE TABLE IF NOT EXISTS `batch_updates` (
  `batch_id` BIGINT NOT NULL,
  `update_id` VARCHAR(40) NOT NULL,
  `start_job_id` INT,
  `n_jobs` INT NOT NULL,
  `committed` BOOLEAN NOT NULL DEFAULT FALSE,
  `time_created` BIGINT,
  `time_committed` BIGINT,
  PRIMARY KEY (`batch_id`, `update_id`)
) ENGINE = InnoDB;
CREATE INDEX `batch_updates_committed` ON `batch_updates` (`batch_id`, `committed`);
CREATE INDEX `batch_updates_start_job_id` ON `batch_updates` (`batch_id`, `start_job_id`);

ALTER TABLE batches ADD COLUMN update_added BOOLEAN DEFAULT FALSE, ALGORITHM=INSTANT;
ALTER TABLE batches_inst_coll_staging ADD COLUMN update_id VARCHAR(40), ALGORITHM=INSTANT;
ALTER TABLE batch_inst_coll_cancellable_resources ADD COLUMN update_id VARCHAR(40), ALGORITHM=INSTANT;

DELIMITER $$

DROP TRIGGER IF EXISTS batches_before_insert $$
CREATE TRIGGER batches_before_insert BEFORE INSERT ON batches
FOR EACH ROW
BEGIN
  DECLARE cur_update_id VARCHAR(40);
  DECLARE new_update_id VARCHAR(40);

  SELECT update_id INTO cur_update_id FROM batch_updates WHERE batch_id = NEW.id;

  IF cur_update_id IS NULL THEN
    SET new_update_id = LEFT(MD5(RAND()), 8);

    INSERT INTO `batch_updates` (`batch_id`, `update_id`, start_job_id, n_jobs, `committed`, time_created, time_committed)
      VALUES (NEW.id, new_update_id, 1, NEW.n_jobs, NEW.state != 'open', NEW.time_created, NEW.time_closed)
    ON DUPLICATE KEY UPDATE n_jobs = n_jobs;
  END IF;

  SET NEW.update_added = TRUE;
END $$

DROP TRIGGER IF EXISTS batches_before_update $$
CREATE TRIGGER batches_before_update BEFORE UPDATE ON batches
FOR EACH ROW
BEGIN
  SET NEW.update_added = TRUE;
END $$

DROP TRIGGER IF EXISTS batches_after_update $$
CREATE TRIGGER batches_after_update AFTER UPDATE ON batches
FOR EACH ROW
BEGIN
  DECLARE new_update_id VARCHAR(40);

  IF NOT OLD.update_added THEN
    SET new_update_id = LEFT(MD5(RAND()), 8);

    INSERT INTO `batch_updates` (`batch_id`, `update_id`, start_job_id, n_jobs, `committed`, time_created, time_committed)
      VALUES (NEW.id, new_update_id, 1, NEW.n_jobs, NEW.state != 'open', NEW.time_created, NEW.time_closed)
    ON DUPLICATE KEY UPDATE n_jobs = n_jobs;

    UPDATE batches_inst_coll_staging
    SET update_id = new_update_id
    WHERE batch_id = NEW.id;

    UPDATE batch_inst_coll_cancellable_resources
    SET update_id = new_update_id
    WHERE batch_id = NEW.id;
  END IF;
END $$

DROP TRIGGER IF EXISTS batches_inst_coll_staging_before_insert $$
CREATE TRIGGER batches_inst_coll_staging_before_insert BEFORE INSERT ON batches_inst_coll_staging
FOR EACH ROW
BEGIN
  DECLARE cur_update_id VARCHAR(40);

  IF NEW.update_id IS NULL THEN
    SELECT update_id INTO cur_update_id FROM batch_updates WHERE batch_id = NEW.batch_id;
    SET NEW.update_id = cur_update_id;
  END IF;
END $$

DROP TRIGGER IF EXISTS batches_inst_coll_staging_before_update $$
CREATE TRIGGER batches_inst_coll_staging_before_update BEFORE UPDATE ON batches_inst_coll_staging
FOR EACH ROW
BEGIN
  DECLARE cur_update_id VARCHAR(40);

  IF OLD.update_id IS NULL THEN
    SELECT update_id INTO cur_update_id FROM batch_updates WHERE batch_id = NEW.batch_id;
    SET NEW.update_id = cur_update_id;
  END IF;
END $$

DROP TRIGGER IF EXISTS batch_inst_coll_cancellable_resources_before_insert $$
CREATE TRIGGER batch_inst_coll_cancellable_resources_before_insert BEFORE INSERT ON batch_inst_coll_cancellable_resources
FOR EACH ROW
BEGIN
  DECLARE cur_update_id VARCHAR(40);

  IF NEW.update_id IS NULL THEN
    SELECT update_id INTO cur_update_id FROM batch_updates WHERE batch_id = NEW.batch_id;
    SET NEW.update_id = cur_update_id;
  END IF;
END $$

DROP TRIGGER IF EXISTS batch_inst_coll_cancellable_resources_before_update $$
CREATE TRIGGER batch_inst_coll_cancellable_resources_before_update BEFORE UPDATE ON batch_inst_coll_cancellable_resources
FOR EACH ROW
BEGIN
  DECLARE cur_update_id VARCHAR(40);

  IF OLD.update_id IS NULL THEN
    SELECT update_id INTO cur_update_id FROM batch_updates WHERE batch_id = NEW.batch_id;
    SET NEW.update_id = cur_update_id;
  END IF;
END $$

DROP PROCEDURE IF EXISTS close_batch $$
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
    FROM batches_inst_coll_staging
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

      UPDATE batch_updates SET `committed` = TRUE, time_committed = in_timestamp
      WHERE batch_id = in_batch_id;

      INSERT INTO user_inst_coll_resources (user, inst_coll, token, n_ready_jobs, ready_cores_mcpu)
      SELECT user, inst_coll, 0, @n_ready_jobs := COALESCE(SUM(n_ready_jobs), 0), @ready_cores_mcpu := COALESCE(SUM(ready_cores_mcpu), 0)
      FROM batches_inst_coll_staging
      JOIN batches ON batches.id = batches_inst_coll_staging.batch_id
      WHERE batch_id = in_batch_id
      GROUP BY `user`, inst_coll
      ON DUPLICATE KEY UPDATE
        n_ready_jobs = n_ready_jobs + @n_ready_jobs,
        ready_cores_mcpu = ready_cores_mcpu + @ready_cores_mcpu;

      DELETE FROM batches_inst_coll_staging WHERE batch_id = in_batch_id;

      COMMIT;
      SELECT 0 as rc;
    ELSE
      ROLLBACK;
      SELECT 2 as rc, expected_n_jobs, staging_n_jobs as actual_n_jobs, 'wrong number of jobs' as message;
    END IF;
  END IF;
END $$

DROP TRIGGER IF EXISTS jobs_after_update $$
CREATE TRIGGER jobs_after_update AFTER UPDATE ON jobs
FOR EACH ROW
BEGIN
  DECLARE cur_user VARCHAR(100);
  DECLARE cur_batch_cancelled BOOLEAN;
  DECLARE cur_n_tokens INT;
  DECLARE cur_update_id VARCHAR(40);
  DECLARE rand_token INT;

  SELECT user INTO cur_user FROM batches WHERE id = NEW.batch_id;

  SELECT update_id INTO cur_update_id
  FROM batch_updates
  WHERE batch_id = NEW.batch_id AND NEW.job_id >= start_job_id AND NEW.job_id < start_job_id + n_jobs;

  SET cur_batch_cancelled = EXISTS (SELECT TRUE
                                    FROM batches_cancelled
                                    WHERE id = NEW.batch_id
                                    LOCK IN SHARE MODE);

  SELECT n_tokens INTO cur_n_tokens FROM globals LOCK IN SHARE MODE;
  SET rand_token = FLOOR(RAND() * cur_n_tokens);

  IF OLD.state = 'Ready' THEN
    IF NOT (OLD.always_run OR OLD.cancelled OR cur_batch_cancelled) THEN
      # cancellable
      INSERT INTO batch_inst_coll_cancellable_resources (batch_id, update_id, inst_coll, token, n_ready_cancellable_jobs, ready_cancellable_cores_mcpu)
      VALUES (OLD.batch_id, cur_update_id, OLD.inst_coll, rand_token, -1, -OLD.cores_mcpu)
      ON DUPLICATE KEY UPDATE
        n_ready_cancellable_jobs = n_ready_cancellable_jobs - 1,
        ready_cancellable_cores_mcpu = ready_cancellable_cores_mcpu - OLD.cores_mcpu;
    END IF;

    IF NOT OLD.always_run AND (OLD.cancelled OR cur_batch_cancelled) THEN
      # cancelled
      INSERT INTO user_inst_coll_resources (user, inst_coll, token, n_cancelled_ready_jobs)
      VALUES (cur_user, OLD.inst_coll, rand_token, -1)
      ON DUPLICATE KEY UPDATE
        n_cancelled_ready_jobs = n_cancelled_ready_jobs - 1;
    ELSE
      # runnable
      INSERT INTO user_inst_coll_resources (user, inst_coll, token, n_ready_jobs, ready_cores_mcpu)
      VALUES (cur_user, OLD.inst_coll, rand_token, -1, -OLD.cores_mcpu)
      ON DUPLICATE KEY UPDATE
        n_ready_jobs = n_ready_jobs - 1,
        ready_cores_mcpu = ready_cores_mcpu - OLD.cores_mcpu;
    END IF;
  ELSEIF OLD.state = 'Running' THEN
    IF NOT (OLD.always_run OR cur_batch_cancelled) THEN
      # cancellable
      INSERT INTO batch_inst_coll_cancellable_resources (batch_id, update_id, inst_coll, token, n_running_cancellable_jobs, running_cancellable_cores_mcpu)
      VALUES (OLD.batch_id, cur_update_id, OLD.inst_coll, rand_token, -1, -OLD.cores_mcpu)
      ON DUPLICATE KEY UPDATE
        n_running_cancellable_jobs = n_running_cancellable_jobs - 1,
        running_cancellable_cores_mcpu = running_cancellable_cores_mcpu - OLD.cores_mcpu;
    END IF;

    # state = 'Running' jobs cannot be cancelled at the job level
    IF NOT OLD.always_run AND cur_batch_cancelled THEN
      # cancelled
      INSERT INTO user_inst_coll_resources (user, inst_coll, token, n_cancelled_running_jobs)
      VALUES (cur_user, OLD.inst_coll, rand_token, -1)
      ON DUPLICATE KEY UPDATE
        n_cancelled_running_jobs = n_cancelled_running_jobs - 1;
    ELSE
      # running
      INSERT INTO user_inst_coll_resources (user, inst_coll, token, n_running_jobs, running_cores_mcpu)
      VALUES (cur_user, OLD.inst_coll, rand_token, -1, -OLD.cores_mcpu)
      ON DUPLICATE KEY UPDATE
        n_running_jobs = n_running_jobs - 1,
        running_cores_mcpu = running_cores_mcpu - OLD.cores_mcpu;
    END IF;
  ELSEIF OLD.state = 'Creating' THEN
    IF NOT (OLD.always_run OR cur_batch_cancelled) THEN
      # cancellable
      INSERT INTO batch_inst_coll_cancellable_resources (batch_id, update_id, inst_coll, token, n_creating_cancellable_jobs)
      VALUES (OLD.batch_id, cur_update_id, OLD.inst_coll, rand_token, -1)
      ON DUPLICATE KEY UPDATE
        n_creating_cancellable_jobs = n_creating_cancellable_jobs - 1;
    END IF;

    # state = 'Creating' jobs cannot be cancelled at the job level
    IF NOT OLD.always_run AND cur_batch_cancelled THEN
      # cancelled
      INSERT INTO user_inst_coll_resources (user, inst_coll, token, n_cancelled_creating_jobs)
      VALUES (cur_user, OLD.inst_coll, rand_token, -1)
      ON DUPLICATE KEY UPDATE
        n_cancelled_creating_jobs = n_cancelled_creating_jobs - 1;
    ELSE
      # creating
      INSERT INTO user_inst_coll_resources (user, inst_coll, token, n_creating_jobs)
      VALUES (cur_user, OLD.inst_coll, rand_token, -1)
      ON DUPLICATE KEY UPDATE
        n_creating_jobs = n_creating_jobs - 1;
    END IF;

  END IF;

  IF NEW.state = 'Ready' THEN
    IF NOT (NEW.always_run OR NEW.cancelled OR cur_batch_cancelled) THEN
      # cancellable
      INSERT INTO batch_inst_coll_cancellable_resources (batch_id, update_id, inst_coll, token, n_ready_cancellable_jobs, ready_cancellable_cores_mcpu)
      VALUES (NEW.batch_id, cur_update_id, NEW.inst_coll, rand_token, 1, NEW.cores_mcpu)
      ON DUPLICATE KEY UPDATE
        n_ready_cancellable_jobs = n_ready_cancellable_jobs + 1,
        ready_cancellable_cores_mcpu = ready_cancellable_cores_mcpu + NEW.cores_mcpu;
    END IF;

    IF NOT NEW.always_run AND (NEW.cancelled OR cur_batch_cancelled) THEN
      # cancelled
      INSERT INTO user_inst_coll_resources (user, inst_coll, token, n_cancelled_ready_jobs)
      VALUES (cur_user, NEW.inst_coll, rand_token, 1)
      ON DUPLICATE KEY UPDATE
        n_cancelled_ready_jobs = n_cancelled_ready_jobs + 1;
    ELSE
      # runnable
      INSERT INTO user_inst_coll_resources (user, inst_coll, token, n_ready_jobs, ready_cores_mcpu)
      VALUES (cur_user, NEW.inst_coll, rand_token, 1, NEW.cores_mcpu)
      ON DUPLICATE KEY UPDATE
        n_ready_jobs = n_ready_jobs + 1,
        ready_cores_mcpu = ready_cores_mcpu + NEW.cores_mcpu;
    END IF;
  ELSEIF NEW.state = 'Running' THEN
    IF NOT (NEW.always_run OR cur_batch_cancelled) THEN
      # cancellable
      INSERT INTO batch_inst_coll_cancellable_resources (batch_id, update_id, inst_coll, token, n_running_cancellable_jobs, running_cancellable_cores_mcpu)
      VALUES (NEW.batch_id, cur_update_id, NEW.inst_coll, rand_token, 1, NEW.cores_mcpu)
      ON DUPLICATE KEY UPDATE
        n_running_cancellable_jobs = n_running_cancellable_jobs + 1,
        running_cancellable_cores_mcpu = running_cancellable_cores_mcpu + NEW.cores_mcpu;
    END IF;

    # state = 'Running' jobs cannot be cancelled at the job level
    IF NOT NEW.always_run AND cur_batch_cancelled THEN
      # cancelled
      INSERT INTO user_inst_coll_resources (user, inst_coll, token, n_cancelled_running_jobs)
      VALUES (cur_user, NEW.inst_coll, rand_token, 1)
      ON DUPLICATE KEY UPDATE
        n_cancelled_running_jobs = n_cancelled_running_jobs + 1;
    ELSE
      # running
      INSERT INTO user_inst_coll_resources (user, inst_coll, token, n_running_jobs, running_cores_mcpu)
      VALUES (cur_user, NEW.inst_coll, rand_token, 1, NEW.cores_mcpu)
      ON DUPLICATE KEY UPDATE
        n_running_jobs = n_running_jobs + 1,
        running_cores_mcpu = running_cores_mcpu + NEW.cores_mcpu;
    END IF;
  ELSEIF NEW.state = 'Creating' THEN
    IF NOT (NEW.always_run OR cur_batch_cancelled) THEN
      # cancellable
      INSERT INTO batch_inst_coll_cancellable_resources (batch_id, update_id, inst_coll, token, n_creating_cancellable_jobs)
      VALUES (NEW.batch_id, cur_update_id, NEW.inst_coll, rand_token, 1)
      ON DUPLICATE KEY UPDATE
        n_creating_cancellable_jobs = n_creating_cancellable_jobs + 1;
    END IF;

    # state = 'Creating' jobs cannot be cancelled at the job level
    IF NOT NEW.always_run AND cur_batch_cancelled THEN
      # cancelled
      INSERT INTO user_inst_coll_resources (user, inst_coll, token, n_cancelled_creating_jobs)
      VALUES (cur_user, NEW.inst_coll, rand_token, 1)
      ON DUPLICATE KEY UPDATE
        n_cancelled_creating_jobs = n_cancelled_creating_jobs + 1;
    ELSE
      # creating
      INSERT INTO user_inst_coll_resources (user, inst_coll, token, n_creating_jobs)
      VALUES (cur_user, NEW.inst_coll, rand_token, 1)
      ON DUPLICATE KEY UPDATE
        n_creating_jobs = n_creating_jobs + 1;
    END IF;
  END IF;
END $$

DROP PROCEDURE IF EXISTS recompute_incremental $$
CREATE PROCEDURE recompute_incremental(
) BEGIN

  DELETE FROM batches_inst_coll_staging;
  DELETE FROM batch_inst_coll_cancellable_resources;
  DELETE FROM user_inst_coll_resources;

  DROP TEMPORARY TABLE IF EXISTS `tmp_batch_inst_coll_resources`;

  CREATE TEMPORARY TABLE `tmp_batch_inst_coll_resources` AS (
    SELECT batch_id, update_id, batch_state, batch_cancelled, user, job_inst_coll,
      COALESCE(SUM(1), 0) as n_jobs,
      COALESCE(SUM(job_state = 'Ready' AND cancellable), 0) as n_ready_cancellable_jobs,
      COALESCE(SUM(IF(job_state = 'Ready' AND cancellable, cores_mcpu, 0)), 0) as ready_cancellable_cores_mcpu,
      COALESCE(SUM(job_state = 'Running' AND cancellable), 0) as n_running_cancellable_jobs,
      COALESCE(SUM(IF(job_state = 'Running' AND cancellable, cores_mcpu, 0)), 0) as running_cancellable_cores_mcpu,
      COALESCE(SUM(job_state = 'Creating' AND cancellable), 0) as n_creating_cancellable_jobs,
      COALESCE(SUM(job_state = 'Running' AND NOT cancelled), 0) as n_running_jobs,
      COALESCE(SUM(IF(job_state = 'Running' AND NOT cancelled, cores_mcpu, 0)), 0) as running_cores_mcpu,
      COALESCE(SUM(job_state = 'Ready' AND runnable), 0) as n_ready_jobs,
      COALESCE(SUM(IF(job_state = 'Ready' AND runnable, cores_mcpu, 0)), 0) as ready_cores_mcpu,
      COALESCE(SUM(job_state = 'Creating' AND NOT cancelled), 0) as n_creating_jobs,
      COALESCE(SUM(job_state = 'Ready' AND cancelled), 0) as n_cancelled_ready_jobs,
      COALESCE(SUM(job_state = 'Running' AND cancelled), 0) as n_cancelled_running_jobs,
      COALESCE(SUM(job_state = 'Creating' AND cancelled), 0) as n_cancelled_creating_jobs
    FROM (
      SELECT batches.user,
        batches.id as batch_id,
        batch_updates.update_id,
        batches.state as batch_state,
        batches.cancelled as batch_cancelled,
        jobs.inst_coll as job_inst_coll,
        jobs.state as job_state,
        jobs.cores_mcpu,
        NOT (jobs.always_run OR jobs.cancelled OR batches.cancelled) AS cancellable,
        (jobs.always_run OR NOT (jobs.cancelled OR batches.cancelled)) AS runnable,
        (NOT jobs.always_run AND (jobs.cancelled OR batches.cancelled)) AS cancelled
      FROM jobs
      INNER JOIN batches
        ON batches.id = jobs.batch_id
      LEFT JOIN batch_updates ON jobs.batch_id = batch_updates.batch_id AND jobs.job_id >= batch_updates.start_job_id
        AND jobs.job_id < batch_updates.start_job_id + batch_updates.n_jobs
      LOCK IN SHARE MODE) as t
    GROUP BY batch_id, update_id, batch_state, batch_cancelled, user, job_inst_coll
  );

  INSERT INTO batches_inst_coll_staging (batch_id, update_id, inst_coll, token, n_jobs, n_ready_jobs, ready_cores_mcpu)
  SELECT batch_id, update_id, job_inst_coll, 0, n_jobs, n_ready_jobs, ready_cores_mcpu
  FROM tmp_batch_inst_coll_resources
  LEFT JOIN batch_updates ON tmp_batch_inst_coll_resources.batch_id = batch_updates.batch_id AND tmp_batch_inst_coll_resources.update_id = batch_updates.update_id
  WHERE NOT `committed`;

  INSERT INTO batch_inst_coll_cancellable_resources (batch_id, update_id, inst_coll, token, n_ready_cancellable_jobs,
    ready_cancellable_cores_mcpu, n_running_cancellable_jobs, running_cancellable_cores_mcpu, n_creating_cancellable_jobs)
  SELECT batch_id, update_id, job_inst_coll, 0, n_ready_cancellable_jobs, ready_cancellable_cores_mcpu,
    n_running_cancellable_jobs, running_cancellable_cores_mcpu, n_creating_cancellable_jobs
  FROM tmp_batch_inst_coll_resources
  LEFT JOIN batch_updates ON tmp_batch_inst_coll_resources.batch_id = batch_updates.batch_id AND tmp_batch_inst_coll_resources.update_id = batch_updates.update_id
  WHERE NOT batch_cancelled AND `committed`;

  INSERT INTO user_inst_coll_resources (user, inst_coll, token, n_ready_jobs, ready_cores_mcpu,
    n_running_jobs, running_cores_mcpu, n_creating_jobs,
    n_cancelled_ready_jobs, n_cancelled_running_jobs, n_cancelled_creating_jobs)
  SELECT t.user, t.job_inst_coll, 0, t.n_ready_jobs, t.ready_cores_mcpu,
    t.n_running_jobs, t.running_cores_mcpu, t.n_creating_jobs,
    t.n_cancelled_ready_jobs, t.n_cancelled_running_jobs, t.n_cancelled_creating_jobs
  FROM (SELECT user, job_inst_coll,
      COALESCE(SUM(n_running_jobs), 0) as n_running_jobs,
      COALESCE(SUM(running_cores_mcpu), 0) as running_cores_mcpu,
      COALESCE(SUM(n_ready_jobs), 0) as n_ready_jobs,
      COALESCE(SUM(ready_cores_mcpu), 0) as ready_cores_mcpu,
      COALESCE(SUM(n_creating_jobs), 0) as n_creating_jobs,
      COALESCE(SUM(n_cancelled_ready_jobs), 0) as n_cancelled_ready_jobs,
      COALESCE(SUM(n_cancelled_running_jobs), 0) as n_cancelled_running_jobs,
      COALESCE(SUM(n_cancelled_creating_jobs), 0) as n_cancelled_creating_jobs
    FROM tmp_batch_inst_coll_resources
    LEFT JOIN batch_updates ON tmp_batch_inst_coll_resources.batch_id = batch_updates.batch_id AND tmp_batch_inst_coll_resources.update_id = batch_updates.update_id
    WHERE batch_state != 'open' AND `committed`
    GROUP by user, job_inst_coll) as t;

  DROP TEMPORARY TABLE IF EXISTS `tmp_batch_inst_coll_resources`;

END $$

DELIMITER ;

COMMIT;

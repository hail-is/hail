START TRANSACTION;

CREATE TABLE IF NOT EXISTS `batch_updates` (
  `batch_id` BIGINT NOT NULL,
  `update_id` INT NOT NULL,
  `token` VARCHAR(100) DEFAULT NULL,
  `start_job_id` INT NOT NULL,
  `n_jobs` INT NOT NULL,
  `committed` BOOLEAN NOT NULL DEFAULT FALSE,
  `time_created` BIGINT,
  `time_committed` BIGINT,
  PRIMARY KEY (`batch_id`, `update_id`),
  FOREIGN KEY (`batch_id`) REFERENCES batches(`id`),
  UNIQUE KEY (`batch_id`, `start_job_id`)
) ENGINE = InnoDB;
CREATE INDEX `batch_updates_committed` ON `batch_updates` (`batch_id`, `committed`);
CREATE INDEX `batch_updates_start_job_id` ON `batch_updates` (`batch_id`, `start_job_id`);

ALTER TABLE batches ADD COLUMN update_added BOOLEAN DEFAULT FALSE, ALGORITHM=INSTANT;
ALTER TABLE jobs ADD COLUMN update_id INT DEFAULT 1, ALGORITHM=INSTANT;
ALTER TABLE batches_inst_coll_staging ADD COLUMN update_id INT DEFAULT 1, ALGORITHM=INSTANT;
ALTER TABLE batch_inst_coll_cancellable_resources ADD COLUMN update_id INT DEFAULT 1, ALGORITHM=INSTANT;

DELIMITER $$

DROP TRIGGER IF EXISTS batches_before_insert $$
CREATE TRIGGER batches_before_insert BEFORE INSERT ON batches
FOR EACH ROW
BEGIN
  SET NEW.update_added = TRUE;
END $$

DROP TRIGGER IF EXISTS batches_after_insert $$
CREATE TRIGGER batches_after_insert AFTER INSERT ON batches
FOR EACH ROW
BEGIN
  IF NEW.n_jobs > 0 THEN
    INSERT INTO `batch_updates` (`batch_id`, `update_id`, `token`, start_job_id, n_jobs, `committed`, time_created)
    VALUES (NEW.id, 1, NEW.token, 1, NEW.n_jobs, FALSE, NEW.time_created);
  END IF;
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
  # We want to change the update to committed when the batch is closed
  IF NEW.n_jobs > 0 THEN
    INSERT INTO `batch_updates` (`batch_id`, `update_id`, `token`, start_job_id, n_jobs, `committed`, time_created, time_committed)
    VALUES (NEW.id, 1, NEW.token, 1, NEW.n_jobs, NEW.state != 'open', NEW.time_created, NEW.time_closed)
    ON DUPLICATE KEY UPDATE committed = NEW.state != 'open';
  END IF;
END $$

DROP TRIGGER IF EXISTS jobs_after_update $$
CREATE TRIGGER jobs_after_update AFTER UPDATE ON jobs
FOR EACH ROW
BEGIN
  DECLARE cur_user VARCHAR(100);
  DECLARE cur_batch_cancelled BOOLEAN;
  DECLARE cur_n_tokens INT;
  DECLARE rand_token INT;

  SELECT user INTO cur_user FROM batches WHERE id = NEW.batch_id;

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
      VALUES (OLD.batch_id, NEW.update_id, OLD.inst_coll, rand_token, -1, -OLD.cores_mcpu)
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
      VALUES (OLD.batch_id, NEW.update_id, OLD.inst_coll, rand_token, -1, -OLD.cores_mcpu)
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
      VALUES (OLD.batch_id, NEW.update_id, OLD.inst_coll, rand_token, -1)
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
      VALUES (NEW.batch_id, NEW.update_id, NEW.inst_coll, rand_token, 1, NEW.cores_mcpu)
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
      VALUES (NEW.batch_id, NEW.update_id, NEW.inst_coll, rand_token, 1, NEW.cores_mcpu)
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
      VALUES (NEW.batch_id, NEW.update_id, NEW.inst_coll, rand_token, 1)
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

DELIMITER ;

COMMIT;

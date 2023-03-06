--  don't need the migrated flag as the billing trigger is only on inserting into attempt_resources
ALTER TABLE attempt_resources ADD COLUMN deduped_resource_id INT DEFAULT NULL, ALGORITHM=INSTANT;

ALTER TABLE aggregated_billing_project_user_resources_v2 ADD COLUMN migrated BOOLEAN DEFAULT FALSE, ALGORITHM=INSTANT;
ALTER TABLE aggregated_billing_project_user_resources_by_date_v2 ADD COLUMN migrated BOOLEAN DEFAULT FALSE, ALGORITHM=INSTANT;
ALTER TABLE aggregated_batch_resources_v2 ADD COLUMN migrated BOOLEAN DEFAULT FALSE, ALGORITHM=INSTANT;
ALTER TABLE aggregated_job_resources_v2 ADD COLUMN migrated BOOLEAN DEFAULT FALSE, ALGORITHM=INSTANT;

DROP TABLE IF EXISTS `aggregated_billing_project_user_resources_v3`;
CREATE TABLE IF NOT EXISTS `aggregated_billing_project_user_resources_v3` (
  `billing_project` VARCHAR(100) NOT NULL,
  `user` VARCHAR(100) NOT NULL,
  `resource_id` INT NOT NULL,
  `token` INT NOT NULL,
  `usage` BIGINT NOT NULL DEFAULT 0,
  PRIMARY KEY (`billing_project`, `user`, `resource_id`, `token`),
  FOREIGN KEY (`billing_project`) REFERENCES billing_projects(name) ON DELETE CASCADE,
  FOREIGN KEY (`resource_id`) REFERENCES resources(`resource_id`) ON DELETE CASCADE
) ENGINE = InnoDB;
CREATE INDEX aggregated_billing_project_user_resources_v3 ON `aggregated_billing_project_user_resources_v3` (`user`);

DROP TABLE IF EXISTS `aggregated_billing_project_user_resources_by_date_v3`;
CREATE TABLE IF NOT EXISTS `aggregated_billing_project_user_resources_by_date_v3` (
  `billing_date` DATE NOT NULL,
  `billing_project` VARCHAR(100) NOT NULL,
  `user` VARCHAR(100) NOT NULL,
  `resource_id` INT NOT NULL,
  `token` INT NOT NULL,
  `usage` BIGINT NOT NULL DEFAULT 0,
  PRIMARY KEY (`billing_date`, `billing_project`, `user`, `resource_id`, `token`),
  FOREIGN KEY (`billing_project`) REFERENCES billing_projects(name) ON DELETE CASCADE,
  FOREIGN KEY (`resource_id`) REFERENCES resources(`resource_id`) ON DELETE CASCADE
) ENGINE = InnoDB;
CREATE INDEX aggregated_billing_project_user_resources_by_date_v3_user ON `aggregated_billing_project_user_resources_by_date_v3` (`billing_date`, `user`);

DROP TABLE IF EXISTS `aggregated_batch_resources_v3`;
CREATE TABLE IF NOT EXISTS `aggregated_batch_resources_v3` (
  `batch_id` BIGINT NOT NULL,
  `resource_id` INT NOT NULL,
  `token` INT NOT NULL,
  `usage` BIGINT NOT NULL DEFAULT 0,
  PRIMARY KEY (`batch_id`, `resource_id`, `token`),
  FOREIGN KEY (`batch_id`) REFERENCES batches(`id`) ON DELETE CASCADE,
  FOREIGN KEY (`resource_id`) REFERENCES resources(`resource_id`) ON DELETE CASCADE
) ENGINE = InnoDB;

DROP TABLE IF EXISTS `aggregated_job_resources_v3`;
CREATE TABLE IF NOT EXISTS `aggregated_job_resources_v3` (
  `batch_id` BIGINT NOT NULL,
  `job_id` INT NOT NULL,
  `resource_id` INT NOT NULL,
  `usage` BIGINT NOT NULL DEFAULT 0,
  PRIMARY KEY (`batch_id`, `job_id`, `resource_id`),
  FOREIGN KEY (`batch_id`) REFERENCES batches(`id`) ON DELETE CASCADE,
  FOREIGN KEY (`batch_id`, `job_id`) REFERENCES jobs(`batch_id`, `job_id`) ON DELETE CASCADE,
  FOREIGN KEY (`resource_id`) REFERENCES resources(`resource_id`) ON DELETE CASCADE
) ENGINE = InnoDB;

DELIMITER $$

-- This is okay here because the deduplication mitigation in batch will have already merged
DROP TRIGGER IF EXISTS attempt_resources_before_insert $$
CREATE TRIGGER attempt_resources_before_insert BEFORE INSERT ON attempt_resources
FOR EACH ROW
BEGIN
  SET NEW.deduped_resource_id = NEW.resource_id;
END $$

DELIMITER $$

DROP TRIGGER IF EXISTS attempts_after_update $$
CREATE TRIGGER attempts_after_update AFTER UPDATE ON attempts
FOR EACH ROW
BEGIN
  DECLARE job_cores_mcpu INT;
  DECLARE cur_billing_project VARCHAR(100);
  DECLARE msec_diff_rollup BIGINT;
  DECLARE cur_n_tokens INT;
  DECLARE rand_token INT;
  DECLARE cur_billing_date DATE;

  SELECT n_tokens INTO cur_n_tokens FROM globals LOCK IN SHARE MODE;
  SET rand_token = FLOOR(RAND() * cur_n_tokens);

  SELECT cores_mcpu INTO job_cores_mcpu FROM jobs
  WHERE batch_id = NEW.batch_id AND job_id = NEW.job_id;

  SELECT billing_project INTO cur_billing_project FROM batches WHERE id = NEW.batch_id;

  SET msec_diff_rollup = (GREATEST(COALESCE(NEW.rollup_time - NEW.start_time, 0), 0) -
                          GREATEST(COALESCE(OLD.rollup_time - OLD.start_time, 0), 0));

  SET cur_billing_date = CAST(UTC_DATE() AS DATE);

  IF msec_diff_rollup != 0 THEN
    INSERT INTO aggregated_billing_project_user_resources_v2 (billing_project, user, resource_id, token, `usage`)
    SELECT billing_project, `user`,
      resource_id,
      rand_token,
      msec_diff_rollup * quantity
    FROM attempt_resources
    JOIN batches ON batches.id = attempt_resources.batch_id
    WHERE batch_id = NEW.batch_id AND job_id = NEW.job_id AND attempt_id = NEW.attempt_id
    ON DUPLICATE KEY UPDATE `usage` = `usage` + msec_diff_rollup * quantity;

    INSERT INTO aggregated_billing_project_user_resources_v3 (billing_project, user, resource_id, token, `usage`)
    SELECT batches.billing_project, batches.`user`,
      attempt_resources.resource_id,
      rand_token,
      msec_diff_rollup * quantity
    FROM attempt_resources
    JOIN batches ON batches.id = attempt_resources.batch_id
    INNER JOIN aggregated_billing_project_user_resources_v2 ON
      aggregated_billing_project_user_resources_v2.billing_project = batches.billing_project AND
      aggregated_billing_project_user_resources_v2.user = batches.user AND
      aggregated_billing_project_user_resources_v2.resource_id = attempt_resources.resource_id AND
      aggregated_billing_project_user_resources_v2.token = rand_token
    WHERE batch_id = NEW.batch_id AND job_id = NEW.job_id AND attempt_id = NEW.attempt_id AND migrated = 1
    ON DUPLICATE KEY UPDATE `usage` = aggregated_billing_project_user_resources_v3.`usage` + msec_diff_rollup * quantity;

    INSERT INTO aggregated_batch_resources_v2 (batch_id, resource_id, token, `usage`)
    SELECT attempt_resources.batch_id,
      resource_id,
      rand_token,
      msec_diff_rollup * quantity
    FROM attempt_resources
    WHERE batch_id = NEW.batch_id AND job_id = NEW.job_id AND attempt_id = NEW.attempt_id
    ON DUPLICATE KEY UPDATE `usage` = `usage` + msec_diff_rollup * quantity;

    INSERT INTO aggregated_batch_resources_v3 (batch_id, resource_id, token, `usage`)
    SELECT attempt_resources.batch_id,
      attempt_resources.resource_id,
      rand_token,
      msec_diff_rollup * quantity
    FROM attempt_resources
    JOIN aggregated_batch_resources_v2 ON
      aggregated_batch_resources_v2.batch_id = attempt_resources.batch_id AND
      aggregated_batch_resources_v2.resource_id = attempt_resources.resource_id AND
      aggregated_batch_resources_v2.token = rand_token
    WHERE batch_id = NEW.batch_id AND job_id = NEW.job_id AND attempt_id = NEW.attempt_id AND migrated = 1
    ON DUPLICATE KEY UPDATE `usage` = aggregated_batch_resources_v3.`usage` + msec_diff_rollup * quantity;

    INSERT INTO aggregated_job_resources_v2 (batch_id, job_id, resource_id, `usage`)
    SELECT attempt_resources.batch_id, attempt_resources.job_id,
      resource_id,
      msec_diff_rollup * quantity
    FROM attempt_resources
    WHERE batch_id = NEW.batch_id AND job_id = NEW.job_id AND attempt_id = NEW.attempt_id
    ON DUPLICATE KEY UPDATE `usage` = `usage` + msec_diff_rollup * quantity;

    INSERT INTO aggregated_job_resources_v3 (batch_id, job_id, resource_id, `usage`)
    SELECT attempt_resources.batch_id, attempt_resources.job_id,
      attempt_resources.resource_id,
      msec_diff_rollup * quantity
    FROM attempt_resources
    JOIN aggregated_job_resources_v2 ON
      aggregated_job_resources_v2.batch_id = attempt_resources.batch_id AND
      aggregated_job_resources_v2.job_id = attempt_resources.job_id AND
      aggregated_job_resources_v2.resource_id = attempt_resources.resource_id AND
      aggregated_job_resources_v2.token = rand_token
    WHERE batch_id = NEW.batch_id AND job_id = NEW.job_id AND attempt_id = NEW.attempt_id AND migrated = 1
    ON DUPLICATE KEY UPDATE `usage` = aggregated_job_resources_v3.`usage` + msec_diff_rollup * quantity;

    INSERT INTO aggregated_billing_project_user_resources_by_date_v2 (billing_date, billing_project, user, resource_id, token, `usage`)
    SELECT cur_billing_date,
      billing_project,
      `user`,
      resource_id,
      rand_token,
      msec_diff_rollup * quantity
    FROM attempt_resources
    JOIN batches ON batches.id = attempt_resources.batch_id
    WHERE batch_id = NEW.batch_id AND job_id = NEW.job_id AND attempt_id = NEW.attempt_id
    ON DUPLICATE KEY UPDATE `usage` = `usage` + msec_diff_rollup * quantity;

    INSERT INTO aggregated_billing_project_user_resources_by_date_v3 (billing_date, billing_project, user, resource_id, token, `usage`)
    SELECT cur_billing_date,
      batches.billing_project,
      batches.`user`,
      attempt_resources.resource_id,
      rand_token,
      msec_diff_rollup * quantity
    FROM attempt_resources
    JOIN batches ON batches.id = attempt_resources.batch_id
    JOIN aggregated_billing_project_user_resources_by_date_v2 ON
      aggregated_billing_project_user_resources_by_date_v2.billing_date = cur_billing_date AND
      aggregated_billing_project_user_resources_by_date_v2.billing_project = batches.billing_project AND
      aggregated_billing_project_user_resources_by_date_v2.user = batches.user AND
      aggregated_billing_project_user_resources_by_date_v2.resource_id = attempt_resources.resource_id AND
      aggregated_billing_project_user_resources_by_date_v2.token = rand_token
    WHERE batch_id = NEW.batch_id AND job_id = NEW.job_id AND attempt_id = NEW.attempt_id AND migrated = 1
    ON DUPLICATE KEY UPDATE `usage` = aggregated_billing_project_user_resources_by_date_v3.`usage` + msec_diff_rollup * quantity;
  END IF;
END $$

DROP TRIGGER IF EXISTS attempt_resources_after_insert $$
CREATE TRIGGER attempt_resources_after_insert AFTER INSERT ON attempt_resources
FOR EACH ROW
BEGIN
  DECLARE cur_start_time BIGINT;
  DECLARE cur_rollup_time BIGINT;
  DECLARE cur_billing_project VARCHAR(100);
  DECLARE cur_user VARCHAR(100);
  DECLARE msec_diff_rollup BIGINT;
  DECLARE cur_n_tokens INT;
  DECLARE rand_token INT;
  DECLARE cur_billing_date DATE;
  DECLARE bp_user_resources_migrated BOOLEAN DEFAULT FALSE;
  DECLARE bp_user_resources_by_date_migrated BOOLEAN DEFAULT FALSE;
  DECLARE batch_resources_migrated BOOLEAN DEFAULT FALSE;
  DECLARE job_resources_migrated BOOLEAN DEFAULT FALSE;

  SELECT billing_project, user INTO cur_billing_project, cur_user
  FROM batches WHERE id = NEW.batch_id;

  SELECT n_tokens INTO cur_n_tokens FROM globals LOCK IN SHARE MODE;
  SET rand_token = FLOOR(RAND() * cur_n_tokens);

  SELECT start_time, rollup_time INTO cur_start_time, cur_rollup_time
  FROM attempts
  WHERE batch_id = NEW.batch_id AND job_id = NEW.job_id AND attempt_id = NEW.attempt_id
  LOCK IN SHARE MODE;

  SET msec_diff_rollup = GREATEST(COALESCE(cur_rollup_time - cur_start_time, 0), 0);

  SET cur_billing_date = CAST(UTC_DATE() AS DATE);

  IF msec_diff_rollup != 0 THEN
    INSERT INTO aggregated_billing_project_user_resources_v2 (billing_project, user, resource_id, token, `usage`)
    VALUES (cur_billing_project, cur_user, NEW.resource_id, rand_token, NEW.quantity * msec_diff_rollup)
    ON DUPLICATE KEY UPDATE
      `usage` = `usage` + NEW.quantity * msec_diff_rollup;

    SELECT migrated INTO bp_user_resources_migrated
    FROM aggregated_billing_project_user_resources_v2
    WHERE billing_project = cur_billing_project AND user = cur_user AND resource_id = NEW.resource_id AND token = rand_token;

    IF bp_user_resources_migrated THEN
      INSERT INTO aggregated_billing_project_user_resources_v3 (billing_project, user, resource_id, token, `usage`)
      VALUES (cur_billing_project, cur_user, NEW.resource_id, rand_token, NEW.quantity * msec_diff_rollup)
      ON DUPLICATE KEY UPDATE
        `usage` = `usage` + NEW.quantity * msec_diff_rollup;
    END IF;

    INSERT INTO aggregated_batch_resources_v2 (batch_id, resource_id, token, `usage`)
    VALUES (NEW.batch_id, NEW.resource_id, rand_token, NEW.quantity * msec_diff_rollup)
    ON DUPLICATE KEY UPDATE
      `usage` = `usage` + NEW.quantity * msec_diff_rollup;

    SELECT migrated INTO batch_resources_migrated
    FROM aggregated_batch_resources_v2
    WHERE batch_id = NEW.batch_id AND resource_id = NEW.resource_id AND token = rand_token;

    IF batch_resources_migrated THEN
      INSERT INTO aggregated_batch_resources_v3 (batch_id, resource_id, token, `usage`)
      VALUES (NEW.batch_id, NEW.resource_id, rand_token, NEW.quantity * msec_diff_rollup)
      ON DUPLICATE KEY UPDATE
        `usage` = `usage` + NEW.quantity * msec_diff_rollup;
    END IF;

    INSERT INTO aggregated_job_resources_v2 (batch_id, job_id, resource_id, `usage`)
    VALUES (NEW.batch_id, NEW.job_id, NEW.resource_id, NEW.quantity * msec_diff_rollup)
    ON DUPLICATE KEY UPDATE
      `usage` = `usage` + NEW.quantity * msec_diff_rollup;

    SELECT migrated INTO job_resources_migrated
    FROM aggregated_job_resources_v2
    WHERE batch_id = NEW.batch_id AND job_id = NEW.job_id AND resource_id = NEW.resource_id AND token = rand_token;

    IF job_resources_migrated THEN
      INSERT INTO aggregated_job_resources_v3 (batch_id, job_id, resource_id, `usage`)
      VALUES (NEW.batch_id, NEW.job_id, NEW.resource_id, NEW.quantity * msec_diff_rollup)
      ON DUPLICATE KEY UPDATE
        `usage` = `usage` + NEW.quantity * msec_diff_rollup;
    END IF;

    INSERT INTO aggregated_billing_project_user_resources_by_date_v2 (billing_date, billing_project, user, resource_id, token, `usage`)
    VALUES (cur_billing_date, cur_billing_project, cur_user, NEW.resource_id, rand_token, NEW.quantity * msec_diff_rollup)
    ON DUPLICATE KEY UPDATE
      `usage` = `usage` + NEW.quantity * msec_diff_rollup;

    SELECT migrated INTO bp_user_resources_by_date_migrated
    FROM aggregated_billing_project_user_resources_by_date_v2
    WHERE billing_date = cur_billing_date AND billing_project = cur_billing_project AND user = cur_user
      AND resource_id = NEW.resource_id AND token = rand_token;

    IF bp_user_resources_by_date_migrated THEN
      INSERT INTO aggregated_billing_project_user_resources_by_date_v3 (billing_date, billing_project, user, resource_id, token, `usage`)
      VALUES (cur_billing_date, cur_billing_project, cur_user, NEW.resource_id, rand_token, NEW.quantity * msec_diff_rollup)
      ON DUPLICATE KEY UPDATE
        `usage` = `usage` + NEW.quantity * msec_diff_rollup;
    END IF;
  END IF;
END $$

DROP TRIGGER IF EXISTS aggregated_bp_user_resources_v2_before_insert $$
CREATE TRIGGER aggregated_bp_user_resources_v2_before_insert BEFORE INSERT ON aggregated_billing_project_user_resources_v2
FOR EACH ROW
BEGIN
  SET NEW.migrated = 1;
END $$

DROP TRIGGER IF EXISTS aggregated_bp_user_resources_by_date_v2_before_insert $$
CREATE TRIGGER aggregated_bp_user_resources_by_date_v2_before_insert BEFORE INSERT ON aggregated_billing_project_user_resources_by_date_v2
FOR EACH ROW
BEGIN
  SET NEW.migrated = 1;
END $$

DROP TRIGGER IF EXISTS aggregated_batch_resources_v2_before_insert $$
CREATE TRIGGER aggregated_batch_resources_v2_before_insert BEFORE INSERT on aggregated_batch_resources_v2
FOR EACH ROW
BEGIN
  SET NEW.migrated = 1;
END $$

DROP TRIGGER IF EXISTS aggregated_job_resources_v2_before_insert $$
CREATE TRIGGER aggregated_job_resources_v2_before_insert BEFORE INSERT on aggregated_job_resources_v2
FOR EACH ROW
BEGIN
  SET NEW.migrated = 1;
END $$

DROP TRIGGER IF EXISTS aggregated_bp_user_resources_v2_after_update $$
CREATE TRIGGER aggregated_bp_user_resources_v2_after_update AFTER UPDATE ON aggregated_billing_project_user_resources_v2
FOR EACH ROW
BEGIN
  DECLARE new_resource_id INT;
  DECLARE cur_n_tokens INT;
  DECLARE rand_token INT;

  IF OLD.migrated = 0 AND NEW.migrated = 1 THEN
    SELECT n_tokens INTO cur_n_tokens FROM globals LOCK IN SHARE MODE;
    SET rand_token = FLOOR(RAND() * cur_n_tokens);

    SELECT deduped_resource_id INTO new_resource_id FROM resources WHERE resource_id = OLD.resource_id;

    INSERT INTO aggregated_billing_project_user_resources_v3 (billing_project, user, resource_id, token, `usage`)
    VALUES (NEW.billing_project, NEW.user, new_resource_id, rand_token, NEW.usage)
    ON DUPLICATE KEY UPDATE
      `usage` = `usage` + NEW.usage;
  END IF;
END $$

DROP TRIGGER IF EXISTS aggregated_bp_user_resources_by_date_v2_after_update $$
CREATE TRIGGER aggregated_bp_user_resources_by_date_v2_after_update AFTER UPDATE ON aggregated_billing_project_user_resources_by_date_v2
FOR EACH ROW
BEGIN
  DECLARE new_resource_id INT;
  DECLARE cur_n_tokens INT;
  DECLARE rand_token INT;

  IF OLD.migrated = 0 AND NEW.migrated = 1 THEN
    SELECT n_tokens INTO cur_n_tokens FROM globals LOCK IN SHARE MODE;
    SET rand_token = FLOOR(RAND() * cur_n_tokens);

    SELECT deduped_resource_id INTO new_resource_id FROM resources WHERE resource_id = OLD.resource_id;

    INSERT INTO aggregated_billing_project_user_resources_by_date_v3 (billing_date, billing_project, user, resource_id, token, `usage`)
    VALUES (NEW.billing_date, NEW.billing_project, NEW.user, new_resource_id, rand_token, NEW.usage)
    ON DUPLICATE KEY UPDATE
        `usage` = `usage` + NEW.usage;
  END IF;
END $$

DROP TRIGGER IF EXISTS aggregated_batch_resources_v2_after_update $$
CREATE TRIGGER aggregated_batch_resources_v2_after_update AFTER UPDATE ON aggregated_batch_resources_v2
FOR EACH ROW
BEGIN
  DECLARE new_resource_id INT;
  DECLARE cur_n_tokens INT;
  DECLARE rand_token INT;

  IF OLD.migrated = 0 AND NEW.migrated = 1 THEN
    SELECT n_tokens INTO cur_n_tokens FROM globals LOCK IN SHARE MODE;
    SET rand_token = FLOOR(RAND() * cur_n_tokens);

    SELECT deduped_resource_id INTO new_resource_id FROM resources WHERE resource_id = OLD.resource_id;

    INSERT INTO aggregated_batch_resources_v3 (batch_id, resource_id, token, `usage`)
    VALUES (NEW.batch_id, new_resource_id, rand_token, NEW.usage)
    ON DUPLICATE KEY UPDATE
      `usage` = `usage` + NEW.usage;
  END IF;
END $$

DROP TRIGGER IF EXISTS aggregated_job_resources_v2_after_update $$
CREATE TRIGGER aggregated_job_resources_v2_after_update AFTER UPDATE ON aggregated_job_resources_v2
FOR EACH ROW
BEGIN
  DECLARE new_resource_id INT;
  DECLARE cur_n_tokens INT;
  DECLARE rand_token INT;

  IF OLD.migrated = 0 AND NEW.migrated = 1 THEN
    SELECT n_tokens INTO cur_n_tokens FROM globals LOCK IN SHARE MODE;
    SET rand_token = FLOOR(RAND() * cur_n_tokens);

    SELECT deduped_resource_id INTO new_resource_id FROM resources WHERE resource_id = OLD.resource_id;

    INSERT INTO aggregated_job_resources_v3 (batch_id, job_id, resource_id, `usage`)
    VALUES (NEW.batch_id, NEW.job_id, new_resource_id, NEW.usage)
    ON DUPLICATE KEY UPDATE
      `usage` = `usage` + NEW.usage;
  END IF;
END $$

DELIMITER ;

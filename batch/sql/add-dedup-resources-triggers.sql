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
    SELECT billing_project, `user`,
      resources.deduped_resource_id,
      rand_token,
      msec_diff_rollup * quantity
    FROM attempt_resources
    JOIN resources ON resources.resource_id = attempt_resources.resource_id
    JOIN batches ON batches.id = attempt_resources.batch_id
    INNER JOIN aggregated_billing_project_user_resources_v2 ON
      aggregated_billing_project_user_resources_v2.billing_project = batches.billing_project AND
      aggregated_billing_project_user_resources_v2.user = batches.user AND
      aggregated_billing_project_user_resources_v2.resource_id = attempt_resources.resource_id AND
      aggregated_billing_project_user_resources_v2.token = rand_token
    WHERE batch_id = NEW.batch_id AND job_id = NEW.job_id AND attempt_id = NEW.attempt_id AND migrated = 1
    ON DUPLICATE KEY UPDATE `usage` = `usage` + msec_diff_rollup * quantity;

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
      resources.deduped_resource_id,
      rand_token,
      msec_diff_rollup * quantity
    FROM attempt_resources
    JOIN resources ON resources.resource_id = attempt_resources.resource_id
    JOIN aggregated_batch_resources_v2 ON
      aggregated_batch_resources_v2.batch_id = attempt_resources.batch_id AND
      aggregated_batch_resources_v2.resource_id = attempt_resources.resource_id AND
      aggregated_batch_resources_v2.token = rand_token
    WHERE batch_id = NEW.batch_id AND job_id = NEW.job_id AND attempt_id = NEW.attempt_id AND migrated = 1
    ON DUPLICATE KEY UPDATE `usage` = `usage` + msec_diff_rollup * quantity;

    INSERT INTO aggregated_job_resources_v2 (batch_id, job_id, resource_id, `usage`)
    SELECT attempt_resources.batch_id, attempt_resources.job_id,
      resource_id,
      msec_diff_rollup * quantity
    FROM attempt_resources
    WHERE batch_id = NEW.batch_id AND job_id = NEW.job_id AND attempt_id = NEW.attempt_id
    ON DUPLICATE KEY UPDATE `usage` = `usage` + msec_diff_rollup * quantity;

    INSERT INTO aggregated_job_resources_v3 (batch_id, job_id, resource_id, `usage`)
    SELECT attempt_resources.batch_id, attempt_resources.job_id,
      resources.deduped_resource_id,
      msec_diff_rollup * quantity
    FROM attempt_resources
    JOIN resources ON resources.resource_id = attempt_resources.resource_id
    JOIN aggregated_job_resources_v2 ON
      aggregated_job_resources_v2.batch_id = attempt_resources.batch_id AND
      aggregated_job_resources_v2.job_id = attempt_resources.job_id AND
      aggregated_job_resources_v2.resource_id = attempt_resources.resource_id AND
      aggregated_job_resources_v2.token = rand_token
    WHERE batch_id = NEW.batch_id AND job_id = NEW.job_id AND attempt_id = NEW.attempt_id AND migrated = 1
    ON DUPLICATE KEY UPDATE `usage` = `usage` + msec_diff_rollup * quantity;

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
      billing_project,
      `user`,
      resources.deduped_resource_id,
      rand_token,
      msec_diff_rollup * quantity
    FROM attempt_resources
    JOIN resources ON resources.resource_id = attempt_resources.resource_id
    JOIN batches ON batches.id = attempt_resources.batch_id
    JOIN aggregated_billing_project_user_resources_by_date_v2 ON
      aggregated_billing_project_user_resources_by_date_v2.billing_date = cur_billing_date AND
      aggregated_billing_project_user_resources_by_date_v2.billing_project = batches.billing_project AND
      aggregated_billing_project_user_resources_by_date_v2.user = batches.user AND
      aggregated_billing_project_user_resources_by_date_v2.resource_id = attempt_resources.resource_id AND
      aggregated_billing_project_user_resources_by_date_v2.token = rand_token
    WHERE batch_id = NEW.batch_id AND job_id = NEW.job_id AND attempt_id = NEW.attempt_id AND migrated = 1
    ON DUPLICATE KEY UPDATE `usage` = `usage` + msec_diff_rollup * quantity;
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
  DECLARE cur_resource VARCHAR(100);
  DECLARE cur_billing_date DATE;
  DECLARE bp_user_resources_migrated BOOLEAN DEFAULT FALSE;
  DECLARE bp_user_resources_by_date_migrated BOOLEAN DEFAULT FALSE;
  DECLARE batch_resources_migrated BOOLEAN DEFAULT FALSE;
  DECLARE job_resources_migrated BOOLEAN DEFAULT FALSE;

  SELECT billing_project, user INTO cur_billing_project, cur_user
  FROM batches WHERE id = NEW.batch_id;

  SELECT n_tokens INTO cur_n_tokens FROM globals LOCK IN SHARE MODE;
  SET rand_token = FLOOR(RAND() * cur_n_tokens);

  SELECT deduped_resource_id INTO cur_resource FROM resources WHERE resource_id = NEW.resource_id;

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
      VALUES (cur_billing_project, cur_user, cur_resource, rand_token, NEW.quantity * msec_diff_rollup)
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
      VALUES (NEW.batch_id, cur_resource, rand_token, NEW.quantity * msec_diff_rollup)
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
      VALUES (NEW.batch_id, NEW.job_id, cur_resource, NEW.quantity * msec_diff_rollup)
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
      VALUES (cur_billing_date, cur_billing_project, cur_user, cur_resource, rand_token, NEW.quantity * msec_diff_rollup)
      ON DUPLICATE KEY UPDATE
        `usage` = `usage` + NEW.quantity * msec_diff_rollup;
    END IF;
  END IF;
END $$

DROP TRIGGER IF EXISTS aggregated_billing_project_user_resources_v2_before_insert $$
CREATE TRIGGER aggregated_billing_project_user_resources_v2_before_insert BEFORE INSERT ON aggregated_billing_project_user_resources_v2
FOR EACH ROW
BEGIN
  SET NEW.migrated = 1;
END $$

DROP TRIGGER IF EXISTS aggregated_billing_project_user_resources_by_date_v2_before_insert $$
CREATE TRIGGER aggregated_billing_project_user_resources_by_date_v2_before_insert BEFORE INSERT ON aggregated_billing_project_user_resources_by_date_v2
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

DROP TRIGGER IF EXISTS aggregated_billing_project_user_resources_v2_after_update $$
CREATE TRIGGER aggregated_billing_project_user_resources_v2_after_update AFTER UPDATE ON aggregated_billing_project_user_resources_v2
FOR EACH ROW
BEGIN
  DECLARE new_resource_id INT;

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

DROP TRIGGER IF EXISTS aggregated_billing_project_user_resources_by_date_v2_after_update $$
CREATE TRIGGER aggregated_billing_project_user_resources_by_date_v2_after_update AFTER UPDATE ON aggregated_billing_project_user_resources_by_date_v2
FOR EACH ROW
BEGIN
  DECLARE new_resource_id INT;

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

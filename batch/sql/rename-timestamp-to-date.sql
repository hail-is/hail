SET autocommit = 0;

START TRANSACTION;

ALTER TABLE aggregated_billing_project_user_resources_by_date_v2 RENAME COLUMN `billing_timestamp` TO `billing_date`;

DELIMITER $$

DROP TRIGGER IF EXISTS attempts_after_update $$
CREATE TRIGGER attempts_after_update AFTER UPDATE ON attempts
FOR EACH ROW
BEGIN
  DECLARE job_cores_mcpu INT;
  DECLARE cur_billing_project VARCHAR(100);
  DECLARE msec_diff BIGINT;
  DECLARE cur_n_tokens INT;
  DECLARE rand_token INT;
  DECLARE cur_billing_date DATE;

  SELECT n_tokens INTO cur_n_tokens FROM globals LOCK IN SHARE MODE;
  SET rand_token = FLOOR(RAND() * cur_n_tokens);

  SELECT cores_mcpu INTO job_cores_mcpu FROM jobs
  WHERE batch_id = NEW.batch_id AND job_id = NEW.job_id;

  SELECT billing_project INTO cur_billing_project FROM batches WHERE id = NEW.batch_id;

  SET msec_diff = (GREATEST(COALESCE(NEW.end_time - NEW.start_time, 0), 0) -
                   GREATEST(COALESCE(OLD.end_time - OLD.start_time, 0), 0));

  IF msec_diff != 0 THEN
    INSERT INTO aggregated_billing_project_resources (billing_project, resource, token, `usage`)
    SELECT billing_project, resources.resource, rand_token, msec_diff * quantity
    FROM attempt_resources
    JOIN batches ON batches.id = attempt_resources.batch_id
    LEFT JOIN resources ON attempt_resources.resource_id = resources.resource_id
    WHERE batch_id = NEW.batch_id AND job_id = NEW.job_id AND attempt_id = NEW.attempt_id
    ON DUPLICATE KEY UPDATE `usage` = `usage` + msec_diff * quantity;

    INSERT INTO aggregated_batch_resources (batch_id, resource, token, `usage`)
    SELECT batch_id, resources.resource, rand_token, msec_diff * quantity
    FROM attempt_resources
    LEFT JOIN resources ON attempt_resources.resource_id = resources.resource_id
    WHERE batch_id = NEW.batch_id AND job_id = NEW.job_id AND attempt_id = NEW.attempt_id
    ON DUPLICATE KEY UPDATE `usage` = `usage` + msec_diff * quantity;

    INSERT INTO aggregated_job_resources (batch_id, job_id, resource, `usage`)
    SELECT batch_id, job_id, resources.resource, msec_diff * quantity
    FROM attempt_resources
    LEFT JOIN resources ON attempt_resources.resource_id = resources.resource_id
    WHERE batch_id = NEW.batch_id AND job_id = NEW.job_id AND attempt_id = NEW.attempt_id
    ON DUPLICATE KEY UPDATE `usage` = `usage` + msec_diff * quantity;

    INSERT INTO aggregated_billing_project_user_resources_v2 (billing_project, user, resource_id, token, `usage`)
    SELECT billing_project, `user`,
      resource_id,
      rand_token,
      msec_diff * quantity
    FROM attempt_resources
    JOIN batches ON batches.id = attempt_resources.batch_id
    WHERE batch_id = NEW.batch_id AND job_id = NEW.job_id AND attempt_id = NEW.attempt_id
    ON DUPLICATE KEY UPDATE `usage` = `usage` + msec_diff * quantity;

    INSERT INTO aggregated_batch_resources_v2 (batch_id, resource_id, token, `usage`)
    SELECT attempt_resources.batch_id,
      resource_id,
      rand_token,
      msec_diff * quantity
    FROM attempt_resources
    WHERE batch_id = NEW.batch_id AND job_id = NEW.job_id AND attempt_id = NEW.attempt_id
    ON DUPLICATE KEY UPDATE `usage` = `usage` + msec_diff * quantity;

    INSERT INTO aggregated_job_resources_v2 (batch_id, job_id, resource_id, `usage`)
    SELECT attempt_resources.batch_id, attempt_resources.job_id,
      resource_id,
      msec_diff * quantity
    FROM attempt_resources
    WHERE batch_id = NEW.batch_id AND job_id = NEW.job_id AND attempt_id = NEW.attempt_id
    ON DUPLICATE KEY UPDATE `usage` = `usage` + msec_diff * quantity;

    IF NEW.end_time IS NOT NULL THEN
      SET cur_billing_date = CAST(FROM_UNIXTIME(NEW.end_time / 1000) AS DATE);

      INSERT INTO aggregated_billing_project_user_resources_by_date_v2 (billing_date, billing_project, user, resource_id, token, `usage`)
      SELECT cur_billing_date,
        billing_project,
        `user`,
        resource_id,
        rand_token,
        msec_diff * quantity
      FROM attempt_resources
      JOIN batches ON batches.id = attempt_resources.batch_id
      WHERE batch_id = NEW.batch_id AND job_id = NEW.job_id AND attempt_id = NEW.attempt_id
      ON DUPLICATE KEY UPDATE `usage` = `usage` + msec_diff * quantity;
    END IF;
  END IF;
END $$

DROP TRIGGER IF EXISTS attempt_resources_after_insert $$
CREATE TRIGGER attempt_resources_after_insert AFTER INSERT ON attempt_resources
FOR EACH ROW
BEGIN
  DECLARE cur_start_time BIGINT;
  DECLARE cur_end_time BIGINT;
  DECLARE cur_billing_project VARCHAR(100);
  DECLARE cur_user VARCHAR(100);
  DECLARE msec_diff BIGINT;
  DECLARE cur_n_tokens INT;
  DECLARE rand_token INT;
  DECLARE cur_resource VARCHAR(100);
  DECLARE cur_billing_date DATE;

  SELECT billing_project, user INTO cur_billing_project, cur_user
  FROM batches WHERE id = NEW.batch_id;

  SELECT n_tokens INTO cur_n_tokens FROM globals LOCK IN SHARE MODE;
  SET rand_token = FLOOR(RAND() * cur_n_tokens);

  SELECT resource INTO cur_resource FROM resources WHERE resource_id = NEW.resource_id;

  SELECT start_time, end_time INTO cur_start_time, cur_end_time
  FROM attempts
  WHERE batch_id = NEW.batch_id AND job_id = NEW.job_id AND attempt_id = NEW.attempt_id
  LOCK IN SHARE MODE;

  SET msec_diff = GREATEST(COALESCE(cur_end_time - cur_start_time, 0), 0);

  SET cur_billing_date = CAST(FROM_UNIXTIME(cur_end_time / 1000) AS DATE);

  IF msec_diff != 0 THEN
    INSERT INTO aggregated_billing_project_resources (billing_project, resource, token, `usage`)
    VALUES (cur_billing_project, cur_resource, rand_token, NEW.quantity * msec_diff)
    ON DUPLICATE KEY UPDATE
      `usage` = `usage` + NEW.quantity * msec_diff;

    INSERT INTO aggregated_batch_resources (batch_id, resource, token, `usage`)
    VALUES (NEW.batch_id, cur_resource, rand_token, NEW.quantity * msec_diff)
    ON DUPLICATE KEY UPDATE
      `usage` = `usage` + NEW.quantity * msec_diff;

    INSERT INTO aggregated_job_resources (batch_id, job_id, resource, `usage`)
    VALUES (NEW.batch_id, NEW.job_id, cur_resource, NEW.quantity * msec_diff)
    ON DUPLICATE KEY UPDATE
      `usage` = `usage` + NEW.quantity * msec_diff;

    INSERT INTO aggregated_billing_project_user_resources_v2 (billing_project, user, resource_id, token, `usage`)
    VALUES (cur_billing_project, cur_user, NEW.resource_id, rand_token, NEW.quantity * msec_diff)
    ON DUPLICATE KEY UPDATE
      `usage` = `usage` + NEW.quantity * msec_diff;

    INSERT INTO aggregated_batch_resources_v2 (batch_id, resource_id, token, `usage`)
    VALUES (NEW.batch_id, NEW.resource_id, rand_token, NEW.quantity * msec_diff)
    ON DUPLICATE KEY UPDATE
      `usage` = `usage` + NEW.quantity * msec_diff;

    INSERT INTO aggregated_job_resources_v2 (batch_id, job_id, resource_id, `usage`)
    VALUES (NEW.batch_id, NEW.job_id, NEW.resource_id, NEW.quantity * msec_diff)
    ON DUPLICATE KEY UPDATE
      `usage` = `usage` + NEW.quantity * msec_diff;

    IF cur_billing_date IS NOT NULL THEN
      INSERT INTO aggregated_billing_project_user_resources_by_date_v2 (billing_date, billing_project, user, resource_id, token, `usage`)
      VALUES (cur_billing_date, cur_billing_project, cur_user, NEW.resource_id, rand_token, NEW.quantity * msec_diff)
      ON DUPLICATE KEY UPDATE
        `usage` = `usage` + NEW.quantity * msec_diff;
    END IF;
  END IF;
END $$

DELIMITER ;

COMMIT;

DELIMITER $$

DROP TRIGGER IF EXISTS attempts_before_update $$
CREATE TRIGGER attempts_before_update BEFORE UPDATE ON attempts
FOR EACH ROW
BEGIN
  IF OLD.start_time IS NOT NULL AND (NEW.start_time IS NULL OR OLD.start_time < NEW.start_time) THEN
    SET NEW.start_time = OLD.start_time;
  END IF;

  # for job private instances that do not finish creating
  IF NEW.reason = 'activation_timeout' THEN
    SET NEW.start_time = NULL;
  END IF;

  IF OLD.reason IS NOT NULL AND (OLD.end_time IS NULL OR NEW.end_time IS NULL OR NEW.end_time >= OLD.end_time) THEN
    SET NEW.end_time = OLD.end_time;
    SET NEW.reason = OLD.reason;
  END IF;
END $$

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

      INSERT INTO aggregated_billing_project_user_resources_by_date_v2 (billing_timestamp, billing_project, user, resource_id, token, `usage`)
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

DELIMITER ;

ALTER TABLE attempts DROP COLUMN migrated, ALGORITHM=INPLACE, LOCK=NONE;

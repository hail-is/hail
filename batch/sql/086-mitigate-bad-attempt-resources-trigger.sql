DELIMITER $$

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
    WHERE billing_project = cur_billing_project AND user = cur_user AND resource_id = NEW.resource_id AND token = rand_token
    FOR UPDATE;

    IF bp_user_resources_migrated THEN
      INSERT INTO aggregated_billing_project_user_resources_v3 (billing_project, user, resource_id, token, `usage`)
      VALUES (cur_billing_project, cur_user, NEW.deduped_resource_id, rand_token, NEW.quantity * msec_diff_rollup)
      ON DUPLICATE KEY UPDATE
        `usage` = `usage` + NEW.quantity * msec_diff_rollup;
    END IF;

    INSERT INTO aggregated_batch_resources_v2 (batch_id, resource_id, token, `usage`)
    VALUES (NEW.batch_id, NEW.resource_id, rand_token, NEW.quantity * msec_diff_rollup)
    ON DUPLICATE KEY UPDATE
      `usage` = `usage` + NEW.quantity * msec_diff_rollup;

    SELECT migrated INTO batch_resources_migrated
    FROM aggregated_batch_resources_v2
    WHERE batch_id = NEW.batch_id AND resource_id = NEW.resource_id AND token = rand_token
    FOR UPDATE;

    IF batch_resources_migrated THEN
      INSERT INTO aggregated_batch_resources_v3 (batch_id, resource_id, token, `usage`)
      VALUES (NEW.batch_id, NEW.deduped_resource_id, rand_token, NEW.quantity * msec_diff_rollup)
      ON DUPLICATE KEY UPDATE
        `usage` = `usage` + NEW.quantity * msec_diff_rollup;
    END IF;

    INSERT INTO aggregated_job_resources_v2 (batch_id, job_id, resource_id, `usage`)
    VALUES (NEW.batch_id, NEW.job_id, NEW.resource_id, NEW.quantity * msec_diff_rollup)
    ON DUPLICATE KEY UPDATE
      `usage` = `usage` + NEW.quantity * msec_diff_rollup;

    SELECT migrated INTO job_resources_migrated
    FROM aggregated_job_resources_v2
    WHERE batch_id = NEW.batch_id AND job_id = NEW.job_id AND resource_id = NEW.resource_id
    FOR UPDATE;

    IF job_resources_migrated THEN
      INSERT INTO aggregated_job_resources_v3 (batch_id, job_id, resource_id, `usage`)
      VALUES (NEW.batch_id, NEW.job_id, NEW.deduped_resource_id, NEW.quantity * msec_diff_rollup)
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
      AND resource_id = NEW.resource_id AND token = rand_token
    FOR UPDATE;

    IF bp_user_resources_by_date_migrated THEN
      INSERT INTO aggregated_billing_project_user_resources_by_date_v3 (billing_date, billing_project, user, resource_id, token, `usage`)
      VALUES (cur_billing_date, cur_billing_project, cur_user, NEW.deduped_resource_id, rand_token, NEW.quantity * msec_diff_rollup)
      ON DUPLICATE KEY UPDATE
        `usage` = `usage` + NEW.quantity * msec_diff_rollup;
    END IF;
  END IF;
END $$

DELIMITER ;

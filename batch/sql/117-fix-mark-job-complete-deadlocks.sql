DELIMITER $$

DROP TRIGGER IF EXISTS attempts_after_update $$
CREATE TRIGGER attempts_after_update AFTER UPDATE ON attempts
FOR EACH ROW
BEGIN
  DECLARE job_cores_mcpu INT;
  DECLARE cur_billing_project VARCHAR(100);
  DECLARE cur_user VARCHAR(100);
  DECLARE msec_diff_rollup BIGINT;
  DECLARE cur_n_tokens INT;
  DECLARE rand_token INT;
  DECLARE cur_billing_date DATE;

  SELECT n_tokens INTO cur_n_tokens FROM globals LOCK IN SHARE MODE;
  SET rand_token = FLOOR(RAND() * cur_n_tokens);

  SELECT cores_mcpu INTO job_cores_mcpu FROM jobs
  WHERE batch_id = NEW.batch_id AND job_id = NEW.job_id;

  SELECT billing_project INTO cur_billing_project FROM batches WHERE id = NEW.batch_id;
  SELECT `user` INTO cur_user FROM batches WHERE id = NEW.batch_id;

  SET msec_diff_rollup = (GREATEST(COALESCE(NEW.rollup_time - NEW.start_time, 0), 0) -
                          GREATEST(COALESCE(OLD.rollup_time - OLD.start_time, 0), 0));

  SET cur_billing_date = CAST(UTC_DATE() AS DATE);

  IF msec_diff_rollup != 0 THEN
    INSERT INTO aggregated_billing_project_user_resources_v3 (billing_project, user, resource_id, token, `usage`)
    SELECT cur_billing_project, cur_user,
      attempt_resources.deduped_resource_id,
      rand_token,
      msec_diff_rollup * quantity
    FROM attempt_resources
    WHERE attempt_resources.batch_id = NEW.batch_id AND attempt_resources.job_id = NEW.job_id AND attempt_id = NEW.attempt_id
    FOR UPDATE
    ON DUPLICATE KEY UPDATE `usage` = aggregated_billing_project_user_resources_v3.`usage` + msec_diff_rollup * quantity;

    INSERT INTO aggregated_job_group_resources_v3 (batch_id, job_group_id, resource_id, token, `usage`)
    SELECT attempt_resources.batch_id,
      job_group_self_and_ancestors.ancestor_id,
      attempt_resources.deduped_resource_id,
      rand_token,
      msec_diff_rollup * quantity
    FROM attempt_resources
    LEFT JOIN jobs ON attempt_resources.batch_id = jobs.batch_id AND attempt_resources.job_id = jobs.job_id
    LEFT JOIN job_group_self_and_ancestors ON jobs.batch_id = job_group_self_and_ancestors.batch_id AND jobs.job_group_id = job_group_self_and_ancestors.job_group_id
    WHERE attempt_resources.batch_id = NEW.batch_id AND attempt_resources.job_id = NEW.job_id AND attempt_resources.attempt_id = NEW.attempt_id
    FOR UPDATE
    ON DUPLICATE KEY UPDATE `usage` = aggregated_job_group_resources_v3.`usage` + msec_diff_rollup * quantity;

    INSERT INTO aggregated_job_resources_v3 (batch_id, job_id, resource_id, `usage`)
    SELECT attempt_resources.batch_id, attempt_resources.job_id,
      attempt_resources.deduped_resource_id,
      msec_diff_rollup * quantity
    FROM attempt_resources
    WHERE attempt_resources.batch_id = NEW.batch_id AND attempt_resources.job_id = NEW.job_id AND attempt_id = NEW.attempt_id
    FOR UPDATE
    ON DUPLICATE KEY UPDATE `usage` = aggregated_job_resources_v3.`usage` + msec_diff_rollup * quantity;

    INSERT INTO aggregated_billing_project_user_resources_by_date_v3 (billing_date, billing_project, user, resource_id, token, `usage`)
    SELECT cur_billing_date,
      cur_billing_project,
      cur_user,
      attempt_resources.deduped_resource_id,
      rand_token,
      msec_diff_rollup * quantity
    FROM attempt_resources
    WHERE attempt_resources.batch_id = NEW.batch_id AND attempt_resources.job_id = NEW.job_id AND attempt_id = NEW.attempt_id
    FOR UPDATE
    ON DUPLICATE KEY UPDATE `usage` = aggregated_billing_project_user_resources_by_date_v3.`usage` + msec_diff_rollup * quantity;
  END IF;
END $$

DELIMITER ;

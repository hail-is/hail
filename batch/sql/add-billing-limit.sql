ALTER TABLE billing_projects ADD `limit` DOUBLE NOT NULL DEFAULT 0;
ALTER TABLE billing_projects ADD `msec_mcpu` BIGINT NOT NULL DEFAULT 0;

CREATE TABLE IF NOT EXISTS `aggregated_billing_project_resources` (
  `billing_project` VARCHAR(100) NOT NULL,
  `resource` VARCHAR(100) NOT NULL,
  `usage` BIGINT NOT NULL DEFAULT 0,
  PRIMARY KEY (`billing_project`, `resource`),
  FOREIGN KEY (`billing_project`) REFERENCES billing_projects(name) ON DELETE CASCADE,
  FOREIGN KEY (`resource`) REFERENCES resources(`resource`) ON DELETE CASCADE
) ENGINE = InnoDB;

DELIMITER $$

DROP TRIGGER IF EXISTS attempts_after_update $$
CREATE TRIGGER attempts_after_update AFTER UPDATE ON attempts
FOR EACH ROW
BEGIN
  DECLARE job_cores_mcpu INT;
  DECLARE cur_billing_project VARCHAR(100);
  DECLARE msec_diff BIGINT;
  DECLARE msec_mcpu_diff BIGINT;

  SELECT cores_mcpu INTO job_cores_mcpu FROM jobs
  WHERE batch_id = NEW.batch_id AND job_id = NEW.job_id;

  SELECT billing_project INTO cur_billing_project FROM batches WHERE id = NEW.batch_id;

  SET msec_diff = (GREATEST(COALESCE(NEW.end_time - NEW.start_time, 0), 0) -
                   GREATEST(COALESCE(OLD.end_time - OLD.start_time, 0), 0));

  SET msec_mcpu_diff = msec_diff * job_cores_mcpu;

  UPDATE batches
  SET msec_mcpu = batches.msec_mcpu + msec_mcpu_diff
  WHERE id = NEW.batch_id;

  UPDATE jobs
  SET msec_mcpu = jobs.msec_mcpu + msec_mcpu_diff
  WHERE batch_id = NEW.batch_id AND job_id = NEW.job_id;

  UPDATE aggregated_billing_project_resources
  JOIN (SELECT billing_project, resource, quantity
        FROM attempt_resources
        LEFT JOIN batches ON batches.id = attempt_resources.batch_id
        WHERE batch_id = NEW.batch_id AND
        job_id = NEW.job_id AND
        attempt_id = NEW.attempt_id) AS t
  ON aggregated_billing_project_resources.billing_project = t.billing_project AND aggregated_billing_project_resources.resource = t.resource
  SET `usage` = `usage` + msec_diff * t.quantity
  WHERE aggregated_billing_project_resources.billing_project = cur_billing_project;

  UPDATE aggregated_batch_resources
  JOIN (SELECT batch_id, resource, quantity
        FROM attempt_resources
        WHERE batch_id = NEW.batch_id AND
        job_id = NEW.job_id AND
        attempt_id = NEW.attempt_id) AS t
  ON aggregated_batch_resources.batch_id = t.batch_id AND
    aggregated_batch_resources.resource = t.resource
  SET `usage` = `usage` + msec_diff * t.quantity
  WHERE aggregated_batch_resources.batch_id = NEW.batch_id;

  UPDATE aggregated_job_resources
  JOIN (SELECT batch_id, job_id, resource, quantity
        FROM attempt_resources
        WHERE batch_id = NEW.batch_id AND
        job_id = NEW.job_id AND
        attempt_id = NEW.attempt_id) AS t
  ON aggregated_job_resources.batch_id = t.batch_id AND
    aggregated_job_resources.job_id = t.job_id AND
    aggregated_job_resources.resource = t.resource
  SET `usage` = `usage` + msec_diff * t.quantity
  WHERE aggregated_job_resources.batch_id = NEW.batch_id AND aggregated_job_resources.job_id = NEW.job_id;
END $$

DROP TRIGGER IF EXISTS attempt_resources_after_insert $$
CREATE TRIGGER attempt_resources_after_insert AFTER INSERT ON attempt_resources
FOR EACH ROW
BEGIN
  DECLARE cur_start_time BIGINT;
  DECLARE cur_end_time BIGINT;
  DECLARE cur_billing_project VARCHAR(100);
  DECLARE msec_diff BIGINT;

  SELECT billing_project INTO cur_billing_project FROM batches WHERE id = NEW.batch_id;

  SELECT start_time, end_time INTO cur_start_time, cur_end_time
  FROM attempts
  WHERE batch_id = NEW.batch_id AND job_id = NEW.job_id AND attempt_id = NEW.attempt_id
  LOCK IN SHARE MODE;

  SET msec_diff = GREATEST(COALESCE(cur_end_time - cur_start_time, 0), 0);

  INSERT INTO aggregated_job_resources (batch_id, job_id, resource, `usage`)
  VALUES (NEW.batch_id, NEW.job_id, NEW.resource, NEW.quantity * msec_diff)
  ON DUPLICATE KEY UPDATE
    `usage` = `usage` + NEW.quantity * msec_diff;

  INSERT INTO aggregated_batch_resources (batch_id, resource, `usage`)
  VALUES (NEW.batch_id, NEW.resource, NEW.quantity * msec_diff)
  ON DUPLICATE KEY UPDATE
    `usage` = `usage` + NEW.quantity * msec_diff;

  INSERT INTO aggregated_billing_project_resources (billing_project, resource, `usage`)
  VALUES (cur_billing_project, NEW.resource, NEW.quantity * msec_diff)
  ON DUPLICATE KEY UPDATE
    `usage` = `usage` + NEW.quantity * msec_diff;
END $$

DELIMITER ;

UPDATE billing_projects
INNER JOIN (
  SELECT billing_project,
  CAST(SUM(IF(format_version < 3, msec_mcpu, 0)) AS SIGNED) as msec_mcpu
  FROM batches
  GROUP BY billing_project) AS t
ON billing_projects.name = t.billing_project
SET billing_projects.msec_mcpu = t.msec_mcpu;

DELETE FROM aggregated_billing_project_resources;

INSERT INTO aggregated_billing_project_resources
(billing_project, resource, `usage`)
SELECT billing_project, resource, SUM(`usage`) as `usage`
FROM aggregated_batch_resources
JOIN batches on aggregated_batch_resources.batch_id = batches.id
GROUP BY billing_project, resource;

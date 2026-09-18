DELIMITER $$

DROP TRIGGER IF EXISTS attempts_before_update;
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

  # does_not_exist means a 404 on VM state -- usually a VM that never booted, but a late
  # poll can also see this for a VM that did run, so guard against nulling billing then.
  IF NEW.reason = 'does_not_exist'
     AND NOT EXISTS (
       SELECT 1 FROM jobs
       WHERE jobs.batch_id = NEW.batch_id
         AND jobs.job_id = NEW.job_id
         AND jobs.attempt_id = NEW.attempt_id
         AND jobs.state = 'Running'
     )
  THEN
    SET NEW.start_time = NULL;
  END IF;

  IF OLD.reason IS NOT NULL AND (OLD.end_time IS NULL OR NEW.end_time IS NULL OR NEW.end_time >= OLD.end_time) THEN
    SET NEW.end_time = OLD.end_time;
    SET NEW.reason = OLD.reason;
  END IF;

  # rollup_time should not go backward in time
  # this could happen if MJS happens after the billing update is received
  IF NEW.rollup_time IS NOT NULL AND OLD.rollup_time IS NOT NULL AND NEW.rollup_time < OLD.rollup_time THEN
    SET NEW.rollup_time = OLD.rollup_time;
  END IF;

  # rollup_time should never be less than the start time
  IF NEW.rollup_time IS NOT NULL AND NEW.start_time IS NOT NULL AND NEW.rollup_time < NEW.start_time THEN
    SET NEW.rollup_time = OLD.rollup_time;
  END IF;

  # rollup_time should never be greater than the end time
  IF NEW.rollup_time IS NOT NULL AND NEW.end_time IS NOT NULL AND NEW.rollup_time > NEW.end_time THEN
    SET NEW.rollup_time = NEW.end_time;
  END IF;
END $$

DELIMITER ;

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

  IF OLD.reason IS NOT NULL AND (OLD.end_time IS NULL OR NEW.end_time IS NULL OR NEW.end_time >= OLD.end_time) THEN
    SET NEW.end_time = OLD.end_time;
    SET NEW.reason = OLD.reason;
  END IF;

  # does_not_exist means the VM returned a 404; null start_time so we don't bill for a VM
  # that never existed, but only if it never activated -- an activated VM is still billed
  IF NEW.reason = 'does_not_exist'
     AND EXISTS (
       SELECT 1 FROM instances
       WHERE instances.name = NEW.instance_name
         AND instances.time_activated IS NULL
     )
  THEN
    SET NEW.start_time = NULL;
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

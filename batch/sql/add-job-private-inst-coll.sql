INSERT INTO inst_colls (`name`, `is_pool`, `boot_disk_size_gb`, `max_instances`, `max_live_instances`)
SELECT 'job-private', 0, boot_disk_size_gb, max_instances, max_live_instances
FROM inst_colls
WHERE `name` = 'standard';

ALTER TABLE instances ADD COLUMN `machine_type` VARCHAR(255);
UPDATE instances SET machine_type = CONCAT('n1-', inst_coll, '-', cores_mcpu DIV 1000);
ALTER TABLE instances MODIFY COLUMN `machine_type` VARCHAR(255) NOT NULL;

ALTER TABLE instances ADD COLUMN `preemptible` BOOLEAN DEFAULT TRUE;
ALTER TABLE instances MODIFY COLUMN `preemptible` BOOLEAN NOT NULL;

CREATE INDEX `instances_time_activated` ON `instances` (`time_activated`);

DELIMITER $$

DROP TRIGGER IF EXISTS attempts_before_update;
CREATE TRIGGER attempts_before_update BEFORE UPDATE ON attempts
FOR EACH ROW
BEGIN
  IF OLD.start_time IS NOT NULL AND (NEW.start_time IS NULL OR OLD.start_time < NEW.start_time) THEN
    SET NEW.start_time = OLD.start_time;
  END IF;

  IF OLD.reason IS NOT NULL AND (OLD.end_time IS NULL OR NEW.end_time IS NULL OR NEW.end_time >= OLD.end_time) THEN
    SET NEW.end_time = OLD.end_time;
    SET NEW.reason = OLD.reason;
  END IF;
END $$

DELIMITER ;
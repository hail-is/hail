CREATE TABLE IF NOT EXISTS `quotes` (
  `id` INT NOT NULL AUTO_INCREMENT,
  `name` VARCHAR(100) NOT NULL,
  `name_cs` VARCHAR(100) NOT NULL COLLATE utf8mb4_0900_as_cs,
  `quote_number` VARCHAR(100) DEFAULT NULL,
  `cost_object` VARCHAR(255) NOT NULL,
  `state` ENUM('open', 'closed') NOT NULL DEFAULT 'open',
  `authorized_amount` DOUBLE DEFAULT NULL,
  `pi_name` VARCHAR(255) DEFAULT NULL,
  `pm_designee` VARCHAR(255) DEFAULT NULL,
  `description` VARCHAR(1000) DEFAULT NULL,
  `time_created` BIGINT NOT NULL,
  PRIMARY KEY (`id`)
) ENGINE = InnoDB;
CREATE UNIQUE INDEX `quote_name` ON `quotes` (`name`);
CREATE UNIQUE INDEX `quote_name_cs` ON `quotes` (`name_cs`);

CREATE TABLE IF NOT EXISTS `quote_managers` (
  `quote_id` INT NOT NULL,
  `user` VARCHAR(100) NOT NULL,
  `role` ENUM('owner', 'manager') NOT NULL DEFAULT 'manager',
  PRIMARY KEY (`quote_id`, `user`),
  FOREIGN KEY (`quote_id`) REFERENCES `quotes`(`id`) ON DELETE CASCADE
) ENGINE = InnoDB;
CREATE INDEX `quote_managers_user` ON `quote_managers` (`user`);

CREATE TABLE IF NOT EXISTS `quote_events` (
  `id` BIGINT NOT NULL AUTO_INCREMENT,
  `quote_id` INT NOT NULL,
  `timestamp` BIGINT NOT NULL,
  `actor` VARCHAR(100) NOT NULL,
  `action` VARCHAR(100) NOT NULL,
  `target_user` VARCHAR(100) DEFAULT NULL,
  `target_project` VARCHAR(100) DEFAULT NULL,
  `detail` JSON DEFAULT NULL,
  `comment` VARCHAR(1000) DEFAULT NULL,
  PRIMARY KEY (`id`),
  FOREIGN KEY (`quote_id`) REFERENCES `quotes`(`id`) ON DELETE CASCADE
) ENGINE = InnoDB;
CREATE INDEX `quote_events_quote_id_timestamp` ON `quote_events` (`quote_id`, `timestamp`);

CREATE TABLE IF NOT EXISTS `billing_project_events` (
  `id` BIGINT NOT NULL AUTO_INCREMENT,
  `billing_project` VARCHAR(100) NOT NULL,
  `timestamp` BIGINT NOT NULL,
  `actor` VARCHAR(100) NOT NULL,
  `action` VARCHAR(100) NOT NULL,
  `target_user` VARCHAR(100) DEFAULT NULL,
  `detail` JSON DEFAULT NULL,
  `comment` VARCHAR(1000) DEFAULT NULL,
  PRIMARY KEY (`id`),
  FOREIGN KEY (`billing_project`) REFERENCES `billing_projects`(`name`) ON DELETE CASCADE
) ENGINE = InnoDB;
CREATE INDEX `billing_project_events_bp_timestamp` ON `billing_project_events` (`billing_project`, `timestamp`);

-- INTERNAL is always id 1. It is the only quote allowed to have a NULL (unlimited) authorized_amount,
-- and the only quote whose billing projects may have a NULL (unlimited) limit. The triggers below
-- enforce this, keyed on the id.
INSERT INTO `quotes` (`id`, `name`, `name_cs`, `cost_object`, `time_created`)
VALUES (1, 'INTERNAL', 'INTERNAL', 'INTERNAL', UNIX_TIMESTAMP() * 1000);

ALTER TABLE `billing_projects`
  ADD COLUMN `quote_id` INT NOT NULL DEFAULT 1,
  ADD COLUMN `description` VARCHAR(1000) DEFAULT NULL,
  ADD CONSTRAINT `fk_billing_projects_quote_id` FOREIGN KEY (`quote_id`) REFERENCES `quotes`(`id`);

-- Database-level guards for the quote / billing project invariants. The application checks these
-- too (with friendlier errors); these triggers make sure a logic error cannot fail open.

DELIMITER $$

DROP PROCEDURE IF EXISTS check_billing_project_invariants $$
CREATE PROCEDURE check_billing_project_invariants(
  IN in_name VARCHAR(100),
  IN in_quote_id INT,
  IN in_status VARCHAR(10),
  IN in_limit DOUBLE
)
BEGIN
  DECLARE cur_quote_id INT DEFAULT NULL;
  DECLARE cur_authorized_amount DOUBLE DEFAULT NULL;
  DECLARE cur_quote_state VARCHAR(10) DEFAULT NULL;
  DECLARE other_limits_sum DOUBLE DEFAULT 0;

  IF in_status != 'deleted' THEN
    IF in_limit IS NOT NULL AND in_limit < 0 THEN
      SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT = 'billing project limit must be non-negative';
    END IF;

    -- Lock the quote row so that concurrent changes to billing projects under the same quote
    -- (and to the quote itself) are serialized against this check.
    SELECT id, authorized_amount, state INTO cur_quote_id, cur_authorized_amount, cur_quote_state
    FROM quotes WHERE id = in_quote_id
    FOR UPDATE;

    -- A missing quote would otherwise look like an unlimited one.
    IF cur_quote_id IS NULL THEN
      SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT = 'billing project quote does not exist';
    END IF;

    IF in_status = 'open' AND cur_quote_state != 'open' THEN
      SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT = 'open billing projects cannot exist under a closed quote';
    END IF;

    IF in_limit IS NULL THEN
      IF in_quote_id != 1 OR cur_authorized_amount IS NOT NULL THEN
        SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT = 'only billing projects under an unlimited INTERNAL quote may be unlimited';
      END IF;
    ELSEIF cur_authorized_amount IS NOT NULL THEN
      SELECT COALESCE(SUM(`limit`), 0) INTO other_limits_sum
      FROM billing_projects
      WHERE quote_id = in_quote_id AND name != in_name AND `status` != 'deleted'
      FOR SHARE;

      IF other_limits_sum + in_limit > cur_authorized_amount THEN
        SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT = 'sum of billing project limits would exceed quote authorized_amount';
      END IF;
    END IF;
  END IF;
END $$

DROP TRIGGER IF EXISTS billing_projects_before_insert $$
CREATE TRIGGER billing_projects_before_insert BEFORE INSERT ON billing_projects
FOR EACH ROW
BEGIN
  CALL check_billing_project_invariants(NEW.name, NEW.quote_id, NEW.`status`, NEW.`limit`);
END $$

DROP TRIGGER IF EXISTS billing_projects_before_update $$
CREATE TRIGGER billing_projects_before_update BEFORE UPDATE ON billing_projects
FOR EACH ROW
BEGIN
  IF NOT (NEW.`limit` <=> OLD.`limit`)
     OR NEW.quote_id != OLD.quote_id
     OR NEW.name != OLD.name
     OR NEW.`status` != OLD.`status` THEN
    CALL check_billing_project_invariants(NEW.name, NEW.quote_id, NEW.`status`, NEW.`limit`);
  END IF;
END $$

DROP TRIGGER IF EXISTS quotes_before_insert $$
CREATE TRIGGER quotes_before_insert BEFORE INSERT ON quotes
FOR EACH ROW
BEGIN
  -- NEW.id is 0 here unless it was given explicitly, so only an explicit id of 1 (INTERNAL) may be unlimited.
  IF NEW.authorized_amount IS NULL AND NEW.id != 1 THEN
    SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT = 'only the INTERNAL quote may be unlimited';
  END IF;
  IF NEW.authorized_amount < 0 THEN
    SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT = 'quote authorized_amount must be non-negative';
  END IF;
END $$

DROP TRIGGER IF EXISTS quotes_before_update $$
CREATE TRIGGER quotes_before_update BEFORE UPDATE ON quotes
FOR EACH ROW
BEGIN
  DECLARE limits_sum DOUBLE DEFAULT 0;
  DECLARE n_unlimited_bps INT DEFAULT 0;
  DECLARE n_open_bps INT DEFAULT 0;

  IF NEW.id != OLD.id THEN
    SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT = 'quote id cannot be changed';
  END IF;
  IF OLD.id = 1 AND (NEW.name != OLD.name OR NEW.name_cs != OLD.name_cs) THEN
    SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT = 'the INTERNAL quote cannot be renamed';
  END IF;
  IF NEW.authorized_amount IS NULL AND NEW.id != 1 THEN
    SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT = 'only the INTERNAL quote may be unlimited';
  END IF;
  IF NEW.authorized_amount < 0 THEN
    SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT = 'quote authorized_amount must be non-negative';
  END IF;

  IF NEW.authorized_amount IS NOT NULL AND NOT (NEW.authorized_amount <=> OLD.authorized_amount) THEN
    SELECT COALESCE(SUM(`limit` IS NULL), 0), COALESCE(SUM(`limit`), 0) INTO n_unlimited_bps, limits_sum
    FROM billing_projects
    WHERE quote_id = NEW.id AND `status` != 'deleted'
    FOR SHARE;

    IF n_unlimited_bps > 0 THEN
      SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT = 'a quote with unlimited billing projects cannot be given an authorized_amount';
    END IF;
    IF limits_sum > NEW.authorized_amount THEN
      SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT = 'quote authorized_amount cannot be less than the sum of its billing project limits';
    END IF;
  END IF;

  IF NEW.state = 'closed' AND OLD.state != 'closed' THEN
    SELECT COUNT(*) INTO n_open_bps
    FROM billing_projects
    WHERE quote_id = NEW.id AND `status` = 'open'
    FOR SHARE;

    IF n_open_bps > 0 THEN
      SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT = 'a quote cannot be closed while it has open billing projects';
    END IF;
  END IF;
END $$

DELIMITER ;

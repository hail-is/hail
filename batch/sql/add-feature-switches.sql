CREATE TABLE IF NOT EXISTS `feature_switches` (
  `compact_billing_tables` BOOLEAN NOT NULL
) ENGINE = InnoDB;

INSERT INTO `feature_switches` (compact_billing_tables) VALUES (0);

CREATE TABLE IF NOT EXISTS `globals` (
  `frozen_merge_deploy` BOOLEAN NOT NULL DEFAULT FALSE
) ENGINE = InnoDB;

INSERT INTO `globals` (frozen_merge_deploy) VALUES (FALSE);

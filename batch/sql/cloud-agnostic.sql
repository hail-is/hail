RENAME TABLE `gevents_mark` TO `events_mark`;
ALTER TABLE instances CHANGE COLUMN `zone` `location` VARCHAR(40) NOT NULL;
ALTER TABLE instances DROP COLUMN worker_config;
ALTER TABLE inst_colls ADD COLUMN `cloud` VARCHAR(100) NOT NULL DEFAULT "gcp";

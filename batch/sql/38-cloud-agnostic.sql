RENAME TABLE `gevents_mark` TO `events_mark`;
ALTER TABLE instances CHANGE COLUMN `zone` `location` VARCHAR(40) NOT NULL;
ALTER TABLE instances CHANGE COLUMN `worker_config` `instance_config` MEDIUMTEXT;
ALTER TABLE inst_colls ADD COLUMN `cloud` VARCHAR(100) NOT NULL DEFAULT "gcp";

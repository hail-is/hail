ALTER TABLE `users` CHANGE COLUMN `email` `login_id` varchar(255) DEFAULT NULL;
ALTER TABLE `users` ADD COLUMN `display_name` varchar(255) DEFAULT NULL;
UPDATE `users` SET `display_name` = `hail_identity`;

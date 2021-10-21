ALTER TABLE `users` CHANGE COLUMN `gsa_key_secret_name` `hail_credentials_secret_name` varchar(255) DEFAULT NULL;

ALTER TABLE `users` ADD COLUMN `hail_identity` varchar(255) DEFAULT NULL;
UPDATE `users` SET `hail_identity` = `gsa_email`;

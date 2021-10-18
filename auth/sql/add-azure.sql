ALTER TABLE `users` ADD COLUMN `azure_application_id` varchar(255) DEFAULT NULL;
ALTER TABLE `users` ADD COLUMN `azure_credentials_secret_name` varchar(255) DEFAULT NULL;
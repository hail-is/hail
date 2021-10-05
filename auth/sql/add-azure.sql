ALTER TABLE `users` ADD COLUMN `azure_service_principal_name` varchar(255) DEFAULT NULL;
ALTER TABLE `users` ADD COLUMN `azure_credentials_secret_name` varchar(255) DEFAULT NULL;

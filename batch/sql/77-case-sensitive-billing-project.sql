ALTER TABLE billing_projects ADD COLUMN name_cs VARCHAR(100) COLLATE utf8mb4_0900_as_cs, ALGORITHM=INSTANT;
ALTER TABLE billing_project_users ADD COLUMN user_cs VARCHAR(100) COLLATE utf8mb4_0900_as_cs, ALGORITHM=INSTANT;

UPDATE billing_projects
SET name_cs = `name`;

UPDATE billing_project_users
SET user_cs = user;

ALTER TABLE billing_projects MODIFY COLUMN name_cs VARCHAR(100) NOT NULL COLLATE utf8mb4_0900_as_cs;
ALTER TABLE billing_project_users MODIFY COLUMN user_cs VARCHAR(100) NOT NULL COLLATE utf8mb4_0900_as_cs;

CREATE UNIQUE INDEX `billing_project_name_cs` ON `billing_projects` (`name_cs`);
CREATE INDEX `billing_project_name_cs_status` ON `billing_projects` (`name_cs`, `status`);
CREATE INDEX `billing_project_users_billing_project_user_cs` ON `billing_project_users` (`billing_project`, `user_cs`);

ALTER TABLE billing_projects ADD COLUMN name_cs VARCHAR(100) COLLATE utf8mb4_0900_as_cs, ALGORITHM=INSTANT;
ALTER TABLE billing_project_users ADD COLUMN user_cs VARCHAR(100) COLLATE utf8mb4_0900_as_cs, ALGORITHM=INSTANT;

UPDATE billing_projects
SET name_cs = `name`;

UPDATE billing_project_users
SET user_cs = user;

ALTER TABLE billing_projects MODIFY COLUMN name_cs VARCHAR(100) COLLATE utf8mb4_0900_as_cs;
ALTER TABLE billing_project_users MODIFY COLUMN user_cs VARCHAR(100) COLLATE utf8mb4_0900_as_cs;

ALTER TABLE billing_projects ADD COLUMN status ENUM('open', 'closed', 'deleted') NOT NULL DEFAULT 'open';
CREATE INDEX `billing_project_status` ON `billing_projects` (`status`);

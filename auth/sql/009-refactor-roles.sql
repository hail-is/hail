-- Table to enumerate the system roles:
CREATE TABLE `system_roles` (
  `id` INT(11) NOT NULL AUTO_INCREMENT,
  `name` VARCHAR(255) NOT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `name` (`name`)
) ENGINE=InnoDB;

INSERT INTO `system_roles` (`name`) VALUES ('admin');
INSERT INTO `system_roles` (`name`) VALUES ('developer');
INSERT INTO `system_roles` (`name`) VALUES ('billing_manager');

-- Table to link roles to users
CREATE TABLE `users_system_roles` (
  `user_id` INT(11) NOT NULL,
  `role_id` INT(11) NOT NULL,
  `assigned_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`user_id`, `role_id`),
  FOREIGN KEY (`user_id`) REFERENCES `users` (`id`),
  FOREIGN KEY (`role_id`) REFERENCES `system_roles` (`id`)
);

-- Table to enumerate the system permissions:
CREATE TABLE `system_permissions` (
  `id` INT(11) NOT NULL AUTO_INCREMENT,
  `name` VARCHAR(255) NOT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `name` (`name`)
) ENGINE=InnoDB;

-- User admin
INSERT INTO `system_permissions` (`name`) VALUES ('create_users');
INSERT INTO `system_permissions` (`name`) VALUES ('read_users');
INSERT INTO `system_permissions` (`name`) VALUES ('update_users');
INSERT INTO `system_permissions` (`name`) VALUES ('delete_users');
-- Role admin
INSERT INTO `system_permissions` (`name`) VALUES ('assign_system_roles');
INSERT INTO `system_permissions` (`name`) VALUES ('read_system_roles');
-- Developer environments
INSERT INTO `system_permissions` (`name`) VALUES ('create_developer_environments');
INSERT INTO `system_permissions` (`name`) VALUES ('read_developer_environments');
INSERT INTO `system_permissions` (`name`) VALUES ('update_developer_environments');
INSERT INTO `system_permissions` (`name`) VALUES ('delete_developer_environments');
-- Logging and monitoring
INSERT INTO `system_permissions` (`name`) VALUES ('view_monitoring_dashboards');
-- Billing projects
INSERT INTO `system_permissions` (`name`) VALUES ('create_billing_projects');
INSERT INTO `system_permissions` (`name`) VALUES ('read_all_billing_projects');
INSERT INTO `system_permissions` (`name`) VALUES ('update_all_billing_projects');
INSERT INTO `system_permissions` (`name`) VALUES ('delete_all_billing_projects');
INSERT INTO `system_permissions` (`name`) VALUES ('assign_users_to_all_billing_projects');

-- Table to link permissions to roles:
CREATE TABLE `system_role_permissions` (
  `role_id` INT(11) NOT NULL,
  `permission_id` INT(11) NOT NULL,
  PRIMARY KEY (`role_id`, `permission_id`),
  FOREIGN KEY (`role_id`) REFERENCES `system_roles` (`id`),
  FOREIGN KEY (`permission_id`) REFERENCES `system_permissions` (`id`)
) ENGINE=InnoDB;

-- Create mapping of system roles to system permissions:
-- User admin
INSERT INTO `system_role_permissions` (`role_id`, `permission_id`) 
SELECT r.id, p.id FROM `system_roles` r, `system_permissions` p 
WHERE r.name = 'admin' AND p.name = 'create_users';

INSERT INTO `system_role_permissions` (`role_id`, `permission_id`) 
SELECT r.id, p.id FROM `system_roles` r, `system_permissions` p 
WHERE r.name = 'admin' AND p.name = 'read_users';

INSERT INTO `system_role_permissions` (`role_id`, `permission_id`) 
SELECT r.id, p.id FROM `system_roles` r, `system_permissions` p 
WHERE r.name = 'admin' AND p.name = 'update_users';

INSERT INTO `system_role_permissions` (`role_id`, `permission_id`) 
SELECT r.id, p.id FROM `system_roles` r, `system_permissions` p 
WHERE r.name = 'admin' AND p.name = 'delete_users';

-- Role admin
INSERT INTO `system_role_permissions` (`role_id`, `permission_id`) 
SELECT r.id, p.id FROM `system_roles` r, `system_permissions` p 
WHERE r.name = 'admin' AND p.name = 'assign_system_roles';

INSERT INTO `system_role_permissions` (`role_id`, `permission_id`) 
SELECT r.id, p.id FROM `system_roles` r, `system_permissions` p 
WHERE r.name = 'admin' AND p.name = 'read_system_roles';

-- Developer environments
INSERT INTO `system_role_permissions` (`role_id`, `permission_id`) 
SELECT r.id, p.id FROM `system_roles` r, `system_permissions` p 
WHERE r.name = 'developer' AND p.name = 'create_developer_environments';

INSERT INTO `system_role_permissions` (`role_id`, `permission_id`) 
SELECT r.id, p.id FROM `system_roles` r, `system_permissions` p 
WHERE r.name = 'developer' AND p.name = 'read_developer_environments';

INSERT INTO `system_role_permissions` (`role_id`, `permission_id`) 
SELECT r.id, p.id FROM `system_roles` r, `system_permissions` p 
WHERE r.name = 'developer' AND p.name = 'update_developer_environments';

INSERT INTO `system_role_permissions` (`role_id`, `permission_id`) 
SELECT r.id, p.id FROM `system_roles` r, `system_permissions` p 
WHERE r.name = 'developer' AND p.name = 'delete_developer_environments';

-- Logging and monitoring
INSERT INTO `system_role_permissions` (`role_id`, `permission_id`) 
SELECT r.id, p.id FROM `system_roles` r, `system_permissions` p 
WHERE r.name = 'developer' AND p.name = 'view_monitoring_dashboards';

-- Billing projects
INSERT INTO `system_role_permissions` (`role_id`, `permission_id`) 
SELECT r.id, p.id FROM `system_roles` r, `system_permissions` p 
WHERE r.name = 'billing_manager' AND p.name = 'read_users';

INSERT INTO `system_role_permissions` (`role_id`, `permission_id`) 
SELECT r.id, p.id FROM `system_roles` r, `system_permissions` p 
WHERE r.name = 'billing_manager' AND p.name = 'create_billing_projects';

INSERT INTO `system_role_permissions` (`role_id`, `permission_id`) 
SELECT r.id, p.id FROM `system_roles` r, `system_permissions` p 
WHERE r.name = 'billing_manager' AND p.name = 'read_all_billing_projects';

INSERT INTO `system_role_permissions` (`role_id`, `permission_id`) 
SELECT r.id, p.id FROM `system_roles` r, `system_permissions` p 
WHERE r.name = 'billing_manager' AND p.name = 'update_all_billing_projects';

INSERT INTO `system_role_permissions` (`role_id`, `permission_id`) 
SELECT r.id, p.id FROM `system_roles` r, `system_permissions` p 
WHERE r.name = 'billing_manager' AND p.name = 'delete_all_billing_projects';

INSERT INTO `system_role_permissions` (`role_id`, `permission_id`) 
SELECT r.id, p.id FROM `system_roles` r, `system_permissions` p 
WHERE r.name = 'billing_manager' AND p.name = 'assign_users_to_all_billing_projects';

-- Give all users the developer, billing_manager and admin roles if they currently have the is_developer flag
INSERT INTO `users_system_roles` (`user_id`, `role_id`)
SELECT `id`, (SELECT `id` FROM `system_roles` WHERE `name` = 'developer')
FROM `users`
WHERE `is_developer` = 1 AND `state` = 'active';

INSERT INTO `users_system_roles` (`user_id`, `role_id`)
SELECT `id`, (SELECT `id` FROM `system_roles` WHERE `name` = 'billing_manager')
FROM `users`
WHERE `is_developer` = 1 AND `state` = 'active';

INSERT INTO `users_system_roles` (`user_id`, `role_id`)
SELECT `id`, (SELECT `id` FROM `system_roles` WHERE `name` = 'admin')
FROM `users`
WHERE `is_developer` = 1 AND `state` = 'active';

-- Add new CI and Deployed System permissions to the system_permissions table
INSERT INTO `system_permissions` (`name`) VALUES ('read_ci');
INSERT INTO `system_permissions` (`name`) VALUES ('manage_ci');
INSERT INTO `system_permissions` (`name`) VALUES ('read_deployed_system_state');
INSERT INTO `system_permissions` (`name`) VALUES ('update_deployed_system_state');

-- Assign new CI permissions to the developer role
INSERT INTO `system_role_permissions` (`role_id`, `permission_id`) 
SELECT r.id, p.id FROM `system_roles` r, `system_permissions` p 
WHERE r.name = 'developer' AND p.name = 'read_ci';

INSERT INTO `system_role_permissions` (`role_id`, `permission_id`) 
SELECT r.id, p.id FROM `system_roles` r, `system_permissions` p 
WHERE r.name = 'developer' AND p.name = 'manage_ci';

-- Assign new deployed system state permissions to the admin role
INSERT INTO `system_role_permissions` (`role_id`, `permission_id`) 
SELECT r.id, p.id FROM `system_roles` r, `system_permissions` p 
WHERE r.name = 'admin' AND p.name = 'read_deployed_system_state';

INSERT INTO `system_role_permissions` (`role_id`, `permission_id`) 
SELECT r.id, p.id FROM `system_roles` r, `system_permissions` p 
WHERE r.name = 'admin' AND p.name = 'update_deployed_system_state';

-- Remove monitoring permissions from the developer role:
DELETE FROM `system_role_permissions` 
WHERE `role_id` = (SELECT `id` FROM `system_roles` WHERE `name` = 'developer') 
AND `permission_id` = (SELECT `id` FROM `system_permissions` WHERE `name` = 'view_monitoring_dashboards');

-- Add monitoring permissions to the admin role:
INSERT INTO `system_role_permissions` (`role_id`, `permission_id`) 
SELECT r.id, p.id FROM `system_roles` r, `system_permissions` p 
WHERE r.name = 'admin' AND p.name = 'view_monitoring_dashboards';

-- Remove namespace management permissions from the developer role:
DELETE FROM `system_role_permissions` 
WHERE `role_id` = (SELECT `id` FROM `system_roles` WHERE `name` = 'developer') 
AND `permission_id` IN (SELECT `id` FROM `system_permissions` WHERE `name` like '%_developer_environments%');

-- Add namespace management permissions to the admin role:
INSERT INTO `system_role_permissions` (`role_id`, `permission_id`) 
SELECT r.id, p.id FROM `system_roles` r, `system_permissions` p 
WHERE r.name = 'admin' AND p.name like '%_developer_environments%';

-- Rename the admin role to sysadmin
UPDATE `system_roles` SET `name` = 'sysadmin' WHERE `name` = 'admin';

-- Assign billing manager role to the 'auth' user:
INSERT INTO `users_system_roles` (`user_id`, `role_id`)
SELECT `id`, (SELECT `id` FROM `system_roles` WHERE `name` = 'billing_manager')
FROM `users`
WHERE `username` = 'auth';

-- Create a new sysadmin-readonly role:
INSERT INTO `system_roles` (`name`) VALUES ('sysadmin-readonly');

-- Assign readonly permissions to the sysadmin-readonly role:
INSERT INTO `system_role_permissions` (`role_id`, `permission_id`)
SELECT (SELECT `id` FROM `system_roles` WHERE `name` = 'sysadmin-readonly'), `id`
FROM `system_permissions`
WHERE `name` = 'read_users';

INSERT INTO `system_role_permissions` (`role_id`, `permission_id`)
SELECT (SELECT `id` FROM `system_roles` WHERE `name` = 'sysadmin-readonly'), `id`
FROM `system_permissions`
WHERE `name` = 'read_system_roles';

INSERT INTO `system_role_permissions` (`role_id`, `permission_id`)
SELECT (SELECT `id` FROM `system_roles` WHERE `name` = 'sysadmin-readonly'), `id`
FROM `system_permissions`
WHERE `name` = 'read_developer_environments';

INSERT INTO `system_role_permissions` (`role_id`, `permission_id`)
SELECT (SELECT `id` FROM `system_roles` WHERE `name` = 'sysadmin-readonly'), `id`
FROM `system_permissions`
WHERE `name` = 'read_ci';

INSERT INTO `system_role_permissions` (`role_id`, `permission_id`)
SELECT (SELECT `id` FROM `system_roles` WHERE `name` = 'sysadmin-readonly'), `id`
FROM `system_permissions`
WHERE `name` = 'read_deployed_system_state';


-- Assign sysadmin-readonly role to the 'ci' user:
INSERT INTO `users_system_roles` (`user_id`, `role_id`)
SELECT `id`, (SELECT `id` FROM `system_roles` WHERE `name` = 'sysadmin-readonly')
FROM `users`
WHERE `username` = 'ci';

-- Assign developer role to the 'ci' user:
INSERT INTO `users_system_roles` (`user_id`, `role_id`)
SELECT `id`, (SELECT `id` FROM `system_roles` WHERE `name` = 'developer')
FROM `users`
WHERE `username` = 'ci';

-- Add read_prerendered_jinja2_context permission to the developer role:
INSERT INTO `system_permissions` (`name`) VALUES ('read_prerendered_jinja2_context');

INSERT INTO `system_role_permissions` (`role_id`, `permission_id`)
SELECT (SELECT `id` FROM `system_roles` WHERE `name` = 'developer'), `id`
FROM `system_permissions`
WHERE `name` = 'read_prerendered_jinja2_context';

-- Assign developer role to the 'grafana' user (for view_monitoring_dashboards permission):
INSERT INTO `users_system_roles` (`user_id`, `role_id`)
SELECT `id`, (SELECT `id` FROM `system_roles` WHERE `name` = 'developer')
FROM `users`
WHERE `username` = 'grafana';

-- Add access_developer_environments permission to the developer role:
INSERT INTO `system_permissions` (`name`) VALUES ('access_developer_environments');

INSERT INTO `system_role_permissions` (`role_id`, `permission_id`)
SELECT (SELECT `id` FROM `system_roles` WHERE `name` = 'developer'), `id`
FROM `system_permissions`
WHERE `name` = 'access_developer_environments';

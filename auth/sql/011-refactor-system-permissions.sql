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
AND `permission_id` = (SELECT `id` FROM `system_permissions` WHERE `name` like '%_developer_namespaces%');

-- Add namespace management permissions to the admin role:
INSERT INTO `system_role_permissions` (`role_id`, `permission_id`) 
SELECT r.id, p.id FROM `system_roles` r, `system_permissions` p 
WHERE r.name = 'admin' AND p.name like '%_developer_namespaces%';

-- Rename the admin role to sysadmin
UPDATE `system_roles` SET `name` = 'sysadmin' WHERE `name` = 'admin';

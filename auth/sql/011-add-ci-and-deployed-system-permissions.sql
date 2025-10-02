-- Add new CI and Deployed System permissions to the system_permissions table
INSERT INTO `system_permissions` (`name`) VALUES ('read_ci');
INSERT INTO `system_permissions` (`name`) VALUES ('manage_ci');
INSERT INTO `system_permissions` (`name`) VALUES ('read_deployed_system_state');
INSERT INTO `system_permissions` (`name`) VALUES ('update_deployed_system_state');

-- Assign these permissions to the admin role
INSERT INTO `system_role_permissions` (`role_id`, `permission_id`) 
SELECT r.id, p.id FROM `system_roles` r, `system_permissions` p 
WHERE r.name = 'admin' AND p.name = 'read_ci';

INSERT INTO `system_role_permissions` (`role_id`, `permission_id`) 
SELECT r.id, p.id FROM `system_roles` r, `system_permissions` p 
WHERE r.name = 'admin' AND p.name = 'manage_ci';

INSERT INTO `system_role_permissions` (`role_id`, `permission_id`) 
SELECT r.id, p.id FROM `system_roles` r, `system_permissions` p 
WHERE r.name = 'admin' AND p.name = 'read_deployed_system_state';

INSERT INTO `system_role_permissions` (`role_id`, `permission_id`) 
SELECT r.id, p.id FROM `system_roles` r, `system_permissions` p 
WHERE r.name = 'admin' AND p.name = 'update_deployed_system_state';

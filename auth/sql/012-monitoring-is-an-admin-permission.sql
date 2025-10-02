-- Remove monitoring permissions from the developer role:
DELETE FROM `system_role_permissions` 
WHERE `role_id` = (SELECT `id` FROM `system_roles` WHERE `name` = 'developer') 
AND `permission_id` = (SELECT `id` FROM `system_permissions` WHERE `name` = 'view_monitoring_dashboards');

-- Add monitoring permissions to the admin role:
INSERT INTO `system_role_permissions` (`role_id`, `permission_id`) 
SELECT r.id, p.id FROM `system_roles` r, `system_permissions` p 
WHERE r.name = 'admin' AND p.name = 'view_monitoring_dashboards';

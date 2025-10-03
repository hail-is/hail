-- Remove namespace management permissions from the developer role:
DELETE FROM `system_role_permissions` 
WHERE `role_id` = (SELECT `id` FROM `system_roles` WHERE `name` = 'developer') 
AND `permission_id` = (SELECT `id` FROM `system_permissions` WHERE `name` like '%_developer_namespaces%');

-- Add namespace management permissions to the admin role:
INSERT INTO `system_role_permissions` (`role_id`, `permission_id`) 
SELECT r.id, p.id FROM `system_roles` r, `system_permissions` p 
WHERE r.name = 'admin' AND p.name like '%_developer_namespaces%';

-- Add create_quotes permission to billing_manager role
INSERT INTO `system_permissions` (`name`) VALUES ('create_quotes');

INSERT INTO `system_role_permissions` (`role_id`, `permission_id`)
SELECT r.id, p.id FROM `system_roles` r, `system_permissions` p
WHERE r.name = 'billing_manager' AND p.name = 'create_quotes';

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
INSERT INTO `system_role_permissions` (`role_id`, `permission_id`)
SELECT (SELECT `id` FROM `system_roles` WHERE `name` = 'developer'), `id`
FROM `system_permissions`
WHERE `name` = 'access_developer_environments';

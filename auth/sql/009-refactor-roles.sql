-- Clear any contents from the current roles table and start again, in case someone has played around with the roles UI or manually edited the database:
CREATE TABLE `system_roles` (
  `id` INT(11) NOT NULL AUTO_INCREMENT,
  `name` VARCHAR(255) NOT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `name` (`name`)
) ENGINE=InnoDB;

INSERT INTO `system_roles` (`name`) VALUES ('developer');
INSERT INTO `system_roles` (`name`) VALUES ('billing_manager');

-- Table to link roles to users
CREATE TABLE `users_system_roles` (
  `user_id` INT(11) NOT NULL,
  `role_id` INT(11) NOT NULL,
  `created_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`user_id`, `role_id`),
  FOREIGN KEY (`user_id`) REFERENCES `users` (`id`),
  FOREIGN KEY (`role_id`) REFERENCES `system_roles` (`id`)
);

-- Give all users the developer role if they currently have the is_developer flag
INSERT INTO `users_system_roles` (`user_id`, `role_id`)
SELECT `id`, (SELECT `id` FROM `system_roles` WHERE `name` = 'developer')
FROM `users`
WHERE `is_developer` = 1;

-- Also give all users the billing_manager role if they currently have the is_developer flag
INSERT INTO `users_system_roles` (`user_id`, `role_id`)
SELECT `id`, (SELECT `id` FROM `system_roles` WHERE `name` = 'billing_manager')
FROM `users`
WHERE `is_developer` = 1;


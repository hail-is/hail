CREATE TABLE `users` (
  `id` INT(11) NOT NULL AUTO_INCREMENT,
  `state` VARCHAR(100) NOT NULL,
  -- creating, active, deleting, deleted
  `username` varchar(255) NOT NULL COLLATE utf8mb4_0900_as_cs,
  `login_id` varchar(255) DEFAULT NULL COLLATE utf8mb4_0900_as_cs,
  `display_name` varchar(255) DEFAULT NULL,
  `is_developer` tinyint(1) NOT NULL DEFAULT 0,
  `is_service_account` tinyint(1) NOT NULL DEFAULT 0,
  -- session
  `tokens_secret_name` varchar(255) DEFAULT NULL,
  -- identity
  `hail_identity` varchar(255) DEFAULT NULL,
  `hail_identity_uid` VARCHAR(255) DEFAULT NULL,
  `hail_credentials_secret_name` varchar(255) DEFAULT NULL,
  -- namespace, for developers
  `namespace_name` varchar(255) DEFAULT NULL,
  `trial_bp_name` varchar(300) DEFAULT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `email` (`email`),
  UNIQUE KEY `username` (`username`)
) ENGINE=InnoDB;

CREATE TABLE `sessions` (
  `session_id` VARCHAR(255) NOT NULL,
  `user_id` INT(11) NOT NULL,
  `max_age_secs` INT(11) UNSIGNED DEFAULT NULL,
  `created` TIMESTAMP DEFAULT NOW(),
  PRIMARY KEY (`session_id`),
  FOREIGN KEY (`user_id`) REFERENCES users(id) ON DELETE CASCADE
) ENGINE=InnoDB;

CREATE EVENT `purge_sessions`
    ON SCHEDULE EVERY 5 MINUTE
    ON COMPLETION PRESERVE
    DO
        DELETE FROM sessions
        WHERE (sessions.max_age_secs IS NOT NULL) AND (TIMESTAMPADD(SECOND, max_age_secs, created) < NOW());

CREATE TABLE `copy_paste_tokens` (
  `id` VARCHAR(255) NOT NULL,
  `session_id` VARCHAR(255) NOT NULL,
  `max_age_secs` INT(11) UNSIGNED DEFAULT NOT NULL,
  `created` TIMESTAMP DEFAULT NOW(),
  PRIMARY KEY (`id`),
  FOREIGN KEY (`session_id`) REFERENCES sessions(session_id) ON DELETE CASCADE
) ENGINE=InnoDB;

CREATE EVENT `purge_copy_paste_tokens`
    ON SCHEDULE EVERY 5 MINUTE
    ON COMPLETION PRESERVE
    DO
        DELETE FROM copy_paste_tokens
        WHERE TIMESTAMPADD(SECOND, max_age_secs, created) < NOW();

CREATE TABLE `roles` (
  `id` INT(11) NOT NULL AUTO_INCREMENT,
  `name` VARCHAR(255) NOT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `name` (`name`)
) ENGINE=InnoDB;

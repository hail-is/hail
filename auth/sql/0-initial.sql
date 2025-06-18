CREATE TABLE `users` (
  `id` INT(11) NOT NULL AUTO_INCREMENT,
  `state` VARCHAR(100) NOT NULL,
  -- creating, active, deleting, deleted
  `username` varchar(255) NOT NULL,
  `email` varchar(255) DEFAULT NULL,
  `is_developer` tinyint(1) NOT NULL DEFAULT 0,
  `is_service_account` tinyint(1) NOT NULL DEFAULT 0,
  -- session
  `tokens_secret_name` varchar(255) DEFAULT NULL,
  -- gsa
  `gsa_email` varchar(255) DEFAULT NULL,
  `gsa_key_secret_name` varchar(255) DEFAULT NULL,
  -- bucket
  `bucket_name` varchar(255) DEFAULT NULL,
  -- namespace, for developers
  `namespace_name` varchar(255) DEFAULT NULL,
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

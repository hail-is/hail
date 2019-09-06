CREATE TABLE `user_data` (
  `id` INT(11) NOT NULL AUTO_INCREMENT,
  `username` varchar(255) NOT NULL,
  `user_id` varchar(255) NOT NULL,
  `email` varchar(255) DEFAULT NULL,
  `developer` tinyint(1) DEFAULT NULL,
  `service_account` tinyint(1) DEFAULT NULL,
  `namespace_name` varchar(255) DEFAULT NULL,
  `gsa_email` varchar(255) NOT NULL,
  `ksa_name` varchar(255) DEFAULT NULL,
  `bucket_name` varchar(255) NOT NULL,
  `gsa_key_secret_name` varchar(255) NOT NULL,
  `jwt_secret_name` varchar(255) NOT NULL,
  `sql_user_secret` varchar(255) DEFAULT NULL,
  `sql_admin_secret` varchar(255) DEFAULT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `user_id` (`user_id`),
  KEY `email` (`email`),
  KEY `username` (`username`)
) ENGINE=InnoDB;

CREATE TABLE `sessions` (
  `session_id` VARCHAR(255) NOT NULL,
  `user_id` INT(11) NOT NULL,
  `max_age_secs` INT(11) UNSIGNED DEFAULT NULL,
  `created` TIMESTAMP DEFAULT NOW(),
  PRIMARY KEY (`session_id`),
  FOREIGN KEY (`user_id`) REFERENCES user_data(id) ON DELETE CASCADE
) ENGINE=InnoDB;

CREATE EVENT `purge_sessions`
    ON SCHEDULE EVERY 5 MINUTE
    ON COMPLETION PRESERVE
    DO
        DELETE FROM sessions
        WHERE (sessions.max_age_secs IS NOT NULL) AND (TIMESTAMPADD(SECOND, max_age_secs, created) < NOW());

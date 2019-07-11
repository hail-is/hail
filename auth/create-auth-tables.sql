CREATE TABLE `user_data` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
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
  `session_id` varchar(255) NOT NULL,
  `kind` varchar(255) NOT NULL,
  `user_id` int(11) NOT NULL,
  `max_age_secs` BIGINT DEFAULT NULL,
  `created` TIMESTAMP DEFAULT NOW(),
  PRIMARY KEY (`session_id`),
  FOREIGN KEY (`user_id`) REFERENCES user_data(id) ON DELETE CASCADE
) ENGINE=InnoDB;

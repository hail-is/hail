CREATE TABLE `notebooks` (
  `user_id` varchar(255) NOT NULL,
  `pod_name` varchar(255) NOT NULL,
  `notebook_token` varchar(255) NOT NULL,
  # Scheduling, Initializing, Running, Ready
  `state` varchar(255) NOT NULL,
  `pod_ip` varchar(255),
  `creation_date` TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  `jupyter_token` varchar(255) NOT NULL,
  PRIMARY KEY (`user_id`)
) ENGINE=InnoDB;

CREATE TABLE `workshops` (
  `id` INT(11) NOT NULL AUTO_INCREMENT,
  `name` varchar(255) NOT NULL,
  `image` varchar(255),
  `cpu` varchar(255),
  `memory` varchar(255),
  `password` varchar(255),
  `active` tinyint(1) DEFAULT 0,
  `token` varchar(255),
  PRIMARY KEY (`id`),
  UNIQUE KEY `name` (`name`)
) ENGINE=InnoDB;

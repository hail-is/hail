CREATE TABLE `workshops` (
  `id` INT(11) NOT NULL AUTO_INCREMENT,
  `name` varchar(255) NOT NULL,
  `image` varchar(255),
  `password` varchar(255),
  `active` tinyint(1) DEFAULT 0,
  `token` varchar(255),
  PRIMARY KEY (`id`),
  UNIQUE KEY `name` (`name`)
) ENGINE=InnoDB;

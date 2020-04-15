CREATE TABLE `copy_paste_tokens` (
  `id` VARCHAR(255) NOT NULL,
  `session_id` VARCHAR(255) NOT NULL,
  `user_id` INT(11) NOT NULL,
  `max_age_secs` INT(11) UNSIGNED NOT NULL,
  `created` TIMESTAMP DEFAULT NOW(),
  PRIMARY KEY (`id`)
) ENGINE=InnoDB;

CREATE EVENT `purge_copy_paste_tokens`
    ON SCHEDULE EVERY 5 MINUTE
    ON COMPLETION PRESERVE
    DO
        DELETE FROM copy_paste_tokens
        WHERE TIMESTAMPADD(SECOND, max_age_secs, created) < NOW();

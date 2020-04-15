DROP EVENT IF EXISTS `purge_sessions`;
DROP EVENT IF EXISTS `purge_copy_paste_tokens`;

DROP TABLE IF EXISTS `copy_paste_tokens`;
DROP TABLE IF EXISTS `sessions`;
DROP TABLE IF EXISTS `users`;
DROP TABLE IF EXISTS `auth_migration_version`;
DROP TABLE IF EXISTS `auth_migrations`;

ALTER TABLE `batches` ADD COLUMN (`token` VARCHAR(100) DEFAULT NULL);
CREATE INDEX `batches_token` ON `batches` (`token`);

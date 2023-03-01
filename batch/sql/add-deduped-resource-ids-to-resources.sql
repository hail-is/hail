ALTER TABLE resources ADD COLUMN deduped_resource_id INT DEFAULT NULL, ALGORITHM=INSTANT;
CREATE INDEX `resources_deduped_resource_id` ON `resources` (`deduped_resource_id`);


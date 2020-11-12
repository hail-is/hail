CREATE TABLE IF NOT EXISTS `atgu_resources` (
  `id` BIGINT NOT NULL AUTO_INCREMENT,
  `time_created` BIGINT NOT NULL,
  `title` TEXT NOT NULL,
  `description` TEXT NOT NULL,
  /* Quill.js documents are stored in the Delta format.  `contents` is
     a JSON-encoded Delta.
     https://quilljs.com/docs/delta/ */
  `contents` LONGTEXT NOT NULL,
  -- comma-separated string of tags
  `tags` TEXT NOT NULL,
  /* `attachments` is a JSON-encoded Dict[str, str] mapping attachment
      IDs to filenames. */
  `attachments` TEXT NOT NULL,
  `time_updated` BIGINT NOT NULL,
  PRIMARY KEY (`id`)
) ENGINE = InnoDB;

const fileResolver = {
  Query: {
    uploads: () => {
      console.info('upload query');
      // Return the record of files uploaded from your DB or API or filesystem.
    }
  },
  Mutation: {
    async uploadFile(parent, { file }) {
      console.info('uploading', file);
      const { stream, filename, mimetype, encoding } = await file;

      console.info(
        'uploaded',
        file,
        // data,
        stream,
        filename,
        mimetype,
        encoding
      );
      // 1. Validate file metadata.

      // 2. Stream file contents into local filesystem or cloud storage:
      // https://nodejs.org/api/stream.html

      // 3. Record the file upload in your DB.
      // const id = await recordFile( … )

      return { stream, filename, mimetype, encoding };
    }
  }
};

module.exports = fileResolver;

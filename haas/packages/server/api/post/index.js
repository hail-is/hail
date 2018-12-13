const fs = require('fs');

const postSchema = fs.readFileSync('./api/post/postSchema.gql', 'utf8');
const postResolver = require('./postResolver');

module.exports = {
  postSchema,
  postResolver
};

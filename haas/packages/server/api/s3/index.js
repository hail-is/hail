const fs = require('fs');

const s3Schema = fs.readFileSync('./api/s3/s3Schema.gql', 'utf8');
const s3Resolver = require('./s3Resolver');

module.exports = {
  s3Schema,
  s3Resolver
};

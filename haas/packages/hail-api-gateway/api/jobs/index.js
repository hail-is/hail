const fs = require('fs');

const jobSchema = fs.readFileSync(
  './api/jobs/graphql/jobs/jobSchema.gql',
  'utf8'
);

const jobResolver = require('./graphql/jobs/jobResolver');

module.exports = {
  jobSchema,
  jobResolver
};

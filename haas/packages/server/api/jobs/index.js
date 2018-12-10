const fs = require('fs');
const JobModels = require('./lib/models');
const JobComm = require('./lib/comm');
const JobRouter = require('./lib/router');
const JobCleaner = require('./lib/retireCron.js');

// const uploadSchema = fs.readFileSync(
//   './api/jobs/graphql/upload/uploadSchema.gql',
//   'utf8'
// );

const jobSchema = fs.readFileSync(
  './api/jobs/graphql/jobs/jobSchema.gql',
  'utf8'
);

const JobResolver = require('./graphql/jobs/jobResolver');

let jF;
const jobsFactory = (comm, user, config) => {
  if (jF) {
    return jF;
  }

  jF = {
    routesRootName: 'jobs'
  };

  const jConfig = config.jobs;
  const jComm = new JobComm(comm, jConfig);
  const jModel = JobModels(jComm, user, jConfig);
  const jobRouter = new JobRouter(jModel, user, jConfig, jComm);
  const jobCleaner = new JobCleaner(jModel);

  // const jobFixer = new JobFixer(jModel);

  jF.router = jobRouter.router; // TODO: this is awkward
  jF.jobModel = jModel;

  jobCleaner.run();
  // jobFixer.run();

  return jF;
};

module.exports = {
  jobSchema,
  JobResolver,
  Jobs: jobsFactory
};

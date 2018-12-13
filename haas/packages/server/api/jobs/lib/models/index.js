const mongoose = require.main.require('./common/database');
const schemas = require('./schemas');
const submissionPlugin = require('./plugins/submission');
const indexCommPlugin = require('./plugins/search/indexSubmission');
const userPlugin = require('./plugins/user');
const deletionPlugin = require('./plugins/delete');

let modelInstance;
module.exports = function jMfactory(jCommInstance, userInstance, jConfig) {
  if (modelInstance) {
    return modelInstance;
  }

  if (!(userInstance && jCommInstance) ) {
    throw new Error('Need user and jobComm instance');
  }

  const jobSchema = schemas.instanceJobSchemaWithOptions(jConfig);

  submissionPlugin.set(jobSchema, {
    modelName: schemas.jobModelName,
    jobComm: jCommInstance
  });

  indexCommPlugin.set(jobSchema, {
    modelName: schemas.jobModelName,
    jobComm: jCommInstance,
    jobSubmissionEvents: submissionPlugin.events
  });

  deletionPlugin.set(jobSchema, {
    modelName: schemas.jobModelName,
    jobComm: jCommInstance,
  });

  userPlugin.set(jobSchema, userInstance, schemas.jobModelName);

  modelInstance = mongoose.model(schemas.jobModelName, jobSchema);

  modelInstance.listenAnnotationEvents();
  modelInstance.listenIndexEvents();
  modelInstance.listenJobExpiration();
  
  return modelInstance;
};

const _ = require('lodash');
const fs = require('fs');
var mongoose = require('mongoose');
var Schema = mongoose.Schema;

const log = require.main.require('./common/logger');

const indexQueue = require.main.require('./api/jobs/lib/indexQueue');

const beanstalkWorker = require.main.require('./api/jobs/lib/beanstalkWorker');

const elasticConfig = Object.assign({}, require.main.require('./common/config').jobs.elastic);


const sqMailer = require.main.require('./common/mail');

const serverAddress = process.env.SERVER_ADDRESS;
const appName = process.env.APP_NAME;

const queueStateSchema = require.main.require('./api/jobs/lib/models/schemas/queueSchema');

const indexSubmissionPlugin = require('./indexSubmission');

if (! (serverAddress && appName) ) {
  throw new Error(`Require process.env to have SERVER_ADDRESS and APP_NAME prop/values`);
}

const lowerCaseAppName = appName.toLowerCase();
// backend and front end konw about this

module.exports.set = function setJobPlugins(Schema, modelName, jobComm, annotationEvents) {
  schema.add({
    // status
    search: {
      indexSubmission: queueStateSchema.get(),
      fieldNames: {type: [String], default: [] },
      indexName: String,
      indexType: String,
      queries: [{}]
    })
  });

  Schema.plugin(indexSubmissionPlugin, {modelName, jobComm, annotationEvents});
};

const _ = require("lodash");
const PromiseBird = require("bluebird");
const fs = PromiseBird.promisifyAll(require("fs-extra"));
const path = require("path");
const Schema = require.main.require("./common/database").Schema;

const log = require.main.require("./common/logger");

const sqMailer = require.main.require("./common/mail");

const serverAddress = process.env.SERVER_ADDRESS;
const appName = process.env.APP_NAME;

const elasticConfig = Object.assign(
  {},
  require.main.require("./common/config").jobs.elastic
);

const elasticsearch = require("elasticsearch");
const elasticClient = new elasticsearch.Client(elasticConfig);

const indexQueueClient = require.main.require("./api/jobs/lib/indexQueue")
  .submitClient;
const annotationQueueClient = require.main.require(
  "./api/jobs/lib/annotationQueue"
).submitClient;

if (!(serverAddress && appName)) {
  throw new Error(
    `Require process.env to have SERVER_ADDRESS and APP_NAME prop/values`
  );
}

const queueStates = require.main.require(
  "./api/jobs/lib/models/schemas/queueSchema"
).states;

const deletionType = queueStates.deleted;
const archiveType = queueStates.archived;

const events = {
  indexDeleted: "indexDeleted",
  indexDeletionFailed: "indexDeletionFailed",
  annotationArchived: "annotationArchived",
  annotationArchiveFailed: "annotationArchiveFailed",
  annotationDeleted: "annotationDeleted",
  annotationDeletionFailed: "annotationDeletionFailed"
};

//TODO: improve logging to web interface. Make all errors log stack trace
module.exports.set = function submissionPlugin(schema, args) {
  const modelName = args.modelName;
  const jobComm = args.jobComm;

  schema.statics.listenJobExpiration = function listenJobExpiration() {
    // TODO: Implement node-cron task to check for job age.
  };

  // TODO: delete only specific index types
  const _deleteIndex = (indexName, cb) => {
    elasticClient.indices.delete(
      {
        index: indexName
      },
      (err, res) => {
        return cb(err);
      }
    );
  };

  const _deleteJobData = jobDoc => {
    return PromiseBird.all([
      PromiseBird.try(() => {
        if (jobDoc.dirs.in && fs.existsSync(jobDoc.dirs.in)) {
          return fs.removeAsync(jobDoc.dirs.in);
        }
      }),
      PromiseBird.try(() => {
        if (jobDoc.dirs.out && fs.existsSync(jobDoc.dirs.out)) {
          return fs.removeAsync(jobDoc.dirs.out);
        }
      })
    ]);
  };

  schema.methods.deleteIndexInQueue = function deleteIndexInQueue(cb) {
    const jobDoc = this;

    // There should be no error besides NOT_FOUND, which is typical if job already completed
    if (
      jobDoc.search &&
      jobDoc.search.activeSubmission &&
      jobDoc.search.activeSubmission.state === "completed"
    ) {
      return cb(null, null);
    }

    if (
      !(
        jobDoc.search.activeSubmission && jobDoc.search.activeSubmission.queueID
      )
    ) {
      return cb(null, null);
    }

    return indexQueueClient.destroy(
      jobDoc.search.activeSubmission.queueID,
      err => {
        if (err === "NOT_FOUND") {
          cb(null, null);
          return;
        }

        cb(err, null);
      }
    );
  };

  // TODO: handle errors from deletion events in beanstalkd
  // Safe errors (partial list):
  // NOT_FOUND (because already consumed)
  schema.methods.deleteOldIndex = function deleteOldIndex(cb) {
    const callback = cb || _.noop;

    // To clarify that "this" refers to the job document (Model instance)
    const jobDoc = this;

    const err = jobDoc.setOldIndexToArchive();

    // TODO: figure out if we ever want to return if err on setOldIndex
    if (err) {
      log.error(err);
    }

    jobDoc.deleteIndexInQueue(err => {
      // TODO: figure out if we ever want to return on error here
      if (err) {
        log.error(err);
        return callback(err, null);
      }

      // but continue, the error relates to the queue, not the search server

      if (!(jobDoc.search && jobDoc.search.indexName)) {
        return callback(null, null);
      }

      jobDoc.set({
        type: archiveType
      });

      _deleteIndex(jobDoc.search.indexName, err => {
        // allow index to be missing, may have been separately deleted
        // we may have already, previously deleted the string
        if (err && err.statusCode != 404) {
          return callback(err);
        }

        //If the job failed due to a validation error, it will never be able to be saved
        //in deleted form, therefore, skip validation
        return jobDoc.save(
          {
            validateBeforeSave: false
          },
          (saveErr, updatedJob) => {
            if (saveErr) {
              log.error(saveErr);
              jobComm.notifyUser(events.failed, jobDoc, jobDoc);
              return callback(saveErr);
            }

            jobComm.notifyUser(
              events.annotationArchived,
              updatedJob,
              updatedJob
            );

            return callback(null, updatedJob);
          }
        );
      });
    });
  };

  // TODO: handle errors from deletion events in beanstalkd
  // Safe errors (partial list):
  // NOT_FOUND (because already consumed)
  schema.methods.deleteAnnotation = function deleteAnnotation(cb) {
    const callback = cb || _.noop;

    // To clarify that "this" refers to the job document (Model instance)
    const jobDoc = this;

    let queue;

    // There should be no error besides NOT_FOUND, which is typical if job already completed
    annotationQueueClient.destroy(jobDoc.submission.queueID, err => {
      if (err != "NOT_FOUND") {
        log.error(err);
      }
    });

    _deleteJobData(jobDoc)
      .then(err => {
        jobDoc.type = deletionType;
        jobDoc.dirs = undefined;

        if (jobDoc.search && jobDoc.search.indexName) {
          _deleteIndex(jobDoc.search.indexName, err => {
            // allow index to be missing, may have been separately deleted
            if (err) {
              log.error(err);
            }
          });

          // There should be no error besides NOT_FOUND, which is typical if job already completed
          if (
            jobDoc.search.activeSubmission &&
            jobDoc.search.activeSubmission.queueID
          ) {
            indexQueueClient.destroy(
              jobDoc.search.activeSubmission.queueID,
              err => {
                if (err != "NOT_FOUND") {
                  log.error(err);
                }
              }
            );
          }
        }

        if (jobDoc.results) {
          // cannot just delete jobDoc.results
          // http://stackoverflow.com/questions/4486926/delete-a-key-from-a-mongodb-document-using-mongoose
          jobDoc.results = undefined;
        }

        //If the job failed due to a validation error, it will never be able to be saved
        //in deleted form, therefore, skip validation
        return jobDoc.save(
          {
            validateBeforeSave: false
          },
          (saveErr, deletedJob) => {
            if (saveErr) {
              log.error(saveErr);
              jobComm.notifyUser(events.failed, jobDoc, jobDoc);
              return callback(saveErr);
            }

            jobComm.notifyUser(
              events.annotationDeleted,
              deletedJob,
              deletedJob
            );

            return callback(null, deletedJob);
          }
        );
      })
      .catch(err => {
        log.error(err);
        jobComm.notifyUser(events.annotationDeletionFailed, jobDoc, jobDoc);

        if (err.code) {
          return callback(err.code);
        }

        return callback(err);
      });
  };

  schema.methods.checkExpiration = function checkExpiration(queryObject, cb) {
    // Implement
  };
};

// //TODO: improve logging to web interface. Make all errors log stack trac

const Schema = require.main.require("./common/database").Schema;
const _util = require("util");
const _ = require("lodash");
const fs = require("fs-extra");
const shortid = require("shortid");

const queueStateSchema = require.main.require(
  "./api/jobs/lib/models/schemas/queueSchema"
);

// 144 years
const jobDefaultExpiration = 144 * 52 * 7 * 24 * 60 * 60 * 1000;

const visibilityStates = ["private", "public"];

const jobModelName = "Job";
module.exports.jobModelName = jobModelName;
module.exports.instanceJobSchemaWithOptions = instanceJobSchemaWithOptions;

const jobTypes = {
  annotation: "annotation",
  searchResult: "searchResult",
  saveFromQuery: "saveFromQuery",
  fork: "annotationFork",
  deleted: "deleted",
  archived: "archived"
};

const jobTypeEnum = Object.keys(jobTypes).map(key => jobTypes[key]);

module.exports.jobTypes = jobTypes;

function instanceJobSchemaWithOptions(passedConfig, passedOpts) {
  const config = passedConfig.schema || {};

  const { baseDir, jobCollection } = config;

  console.info(passedConfig);

  if (!(baseDir && jobCollection)) {
    throw new Error("baseDir and jobCollection must be provided");
  }

  // if umask 022, will get o+wrx g+rx, see README.md
  fs.ensureDir(config.baseDir, "0770", err => {
    if (err) {
      throw new Error(
        `base directory ${config.baseDir}
      could not be created and doesn\'t exist`,
        err
      );
    }
  });

  const outputExtension = config.outputExtension || ".annotated";

  const jobQueueSchema = queueStateSchema.instanceJobQueue();

  const maxQueriesToSave = config.maxQueriesToSave || 100;
  //Nice post about why timestamps is maybe not a good idea
  //Immutable data
  //http://stackoverflow.com/questions/12669615/add-created-at-and-updated-at-fields-to-mongoose-schemas
  //Allows history
  const opts = {
    collection: jobCollection,
    timestamps: true
  };
  if (passedOpts) {
    opts = _.merge(opts, passedOpts);
  }

  const jobSchema = new Schema(
    {
      assembly: {
        type: String,
        required: true
      },
      // TODO: get rid of this, stick with timestamps, this is confusing and redundant
      // and then rely on the underlying _id to query, and check agianst hte userID stored
      // for authentication
      // publicID: { type: String, required: true },
      // This is used as the base name of the output files
      name: {
        type: String,
        required: true
      },
      dirs: {
        // A job may not have this if made from fork of another job (based on query)
        in: String,
        out: {
          type: String,
          required: true
        }
      },
      config: {},
      // null or object
      options: {
        type: Object,
        default: {
          index: true,

          // set to false because user needs to connect their bucket, to ours,
          // using an IAM account and the S3 bucket permission page
          uploadToS3: false
        }
      },
      // A job may be created from a query rather than an inputFilePath
      inputQueryConfig: {
        queryBody: Object,
        indexName: String,
        indexType: String,
        fieldNames: [String],
        // The elasticsearch config
        indexConfig: String,
        // Transformations to apply during save
        pipeline: String
      },
      // null or object
      userID: {
        type: String,
        required: true
      },
      email: String,
      outputFileNames: Object,
      // The results of a submission includes some statistics
      results: Object,

      // Submission for the job to be annotated, gets one queue
      submission: jobQueueSchema,
      archivedSubmissions: [jobQueueSchema],
      exports: {
        // Should have jobQueueSchema, but we want multiple
        // jobs, with only 1 activeSubmission per export type
        activeSubmission: {},
        archivedSubmissions: [jobQueueSchema],
        exportedFiles: {}
      },
      search: {
        activeSubmission: jobQueueSchema,
        // You can submit multiple indexes, because you may want to index
        // with added data
        archivedSubmissions: [jobQueueSchema],
        //output field names / header
        //Every search object should have this, at some point
        fieldNames: [String],
        indexName: String,
        indexType: String,
        // Array of objects, this._schema.caster.cast
        indexConfig: [{}],
        // any queries executed on the annotation are stored
        queries: [
          {
            query: String,
            date: Date
          }
        ],
        synonyms: Object
        // Search results are just jobs, created from a query
        // Therefore they self reference
        // These search jobs are treated just like any other annotation
        // i.e they are stored in the collection, but referenced here
        // saved
        // This is a graph of results
        // These are like forks
        // TODO: should we just get rid of this, and keep only forks?
        // I see forks as
        // savedResults: [{ type : Schema.Types.ObjectId, ref: jobModelName}],
      },

      // May want to reconsider using this; could make the objects immutable
      // instead
      actionsTaken: [
        {
          date: Date,
          action: String,
          resultStatus: String
        }
      ],

      // In the future we'll add "update", "savedResult", etc
      type: {
        type: String,
        enum: jobTypeEnum,
        default: jobTypes.annotation
      },
      visibility: {
        type: String,
        enum: visibilityStates,
        default: "private"
      },
      // userID : linux permission
      sharedWith: Object(),
      notes: String,
      // Can be thought of as "parents"
      // http://mongoosejs.com/docs/guide.html
      _creator: {
        type: Schema.Types.ObjectId,
        ref: jobModelName
      },
      // An annotation has an expiration date
      expireDate: {
        type: Date,
        default: () => {
          return Date.now() + jobDefaultExpiration;
        }
      }
    },
    opts
  );

  jobSchema.index({
    userID: 1,
    _id: 1
  });
  jobSchema.index({
    userID: 1,
    _id: 1,
    "submission.state": 1,
    type: 1
  });
  jobSchema.index({
    _creator: 1
  });
  jobSchema.index({
    type: 1
  });

  jobSchema.set("toJSON", {
    transform: function(doc, ret) {
      //for older jobs, before schema change
      // ret.inputFileName = ret.inputFileName || path.basename(ret.inputFilePath);
      if (ret.inputQueryConfig) {
        delete ret.inputQueryConfig.indexName;
        delete ret.inputQueryConfig.indexType;
      }

      delete ret.outputBasePath;
      delete ret.inputFilePath;
      delete ret.dirs;
      delete ret.__v;
      return ret;
    }
  });

  jobSchema.set("toObject", {
    virtuals: true,
    minimize: false,
    versionKey: false
  });

  jobSchema.statics.maxQueriesToSave = maxQueriesToSave;

  // For graphql
  jobSchema.virtual("id").get(function() {
    return this._id.toString();
  });

  jobSchema.virtual("outputExtension").get(function getOutputExtension() {
    return outputExtension;
  });

  jobSchema.virtual("baseDir").get(function getBaseDir() {
    return baseDir;
  });

  jobSchema.statics.getBaseSchemaFieldNames = function getBaseSchemaFieldNames() {
    return Object.keys(jobSchema);
  };

  // TODO: combine these 2 isPermitted for function; allow calling with a job object
  // (not nec a Mongoose object), as well as with the Jobs class
  jobSchema.statics.isPermittedFor = (userID, job, permission) =>
    isPermittedFor.call(job, userID, permission);

  // TODO: check that this works without needs isPermittedFor.call()
  jobSchema.methods.isPermittedFor = isPermittedFor;

  // requires "this" to have: userID, visibility, sharedWith properties
  function isPermittedFor(userID, permission = "read") {
    let desiredPermission;

    if (isNumeric(permission)) {
      desiredPermission = permission;
    } else {
      if (permission == "read") {
        desiredPermission = 400;
      } else if (permission == "write") {
        desiredPermission = 600;
      }
    }

    if (this.visibility === "public" && desiredPermission === 400) {
      return true;
    }

    // TODO: is a function argument declared ? I think so, if so typeof === 'undefined'
    // equivalent to === undefind
    // https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/undefined
    if (typeof userID === "undefined") {
      return false;
    }

    if (userID === this.userID) {
      return true;
    }

    const userPermission = this.sharedWith && this.sharedWith[userID];

    if (!userPermission) {
      return false;
    }

    return userPermission >= desiredPermission;
  }

  function escapeFilename(val) {
    return val.replace(/[^a-zA-Z0-9_-]/gi, "_").toLowerCase();
  }

  function isNumeric(n) {
    return !isNaN(parseFloat(n)) && isFinite(n);
  }

  return jobSchema;
}

// _util.inherits(CreateSchemas, Schema);

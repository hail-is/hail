const Schema = require.main.require("./common/database").Schema;

const states = {
  notSubmitted: "not_submitted",
  submitted: "submitted",
  started: "started",
  failed: "failed",
  completed: "completed",
  archived: "archived",
  deleted: "deleted",
  gone: "gone"
};

module.exports.states = states;

const base = {
  state: {
    type: String,
    default: states.notSubmitted
  },
  attempts: {
    type: Number,
    default: 0
  }, // how many times job retried
  log: {
    exception: String,
    // How many sites were annotated, indexed, etc
    progress: {
      type: Number,
      default: 0
    },
    // How many input lines were skipped, rather than added to progress
    skipped: {
      type: Number,
      default: 0
    },
    messages: {
      type: [String],
      default: []
    }
  },
  // the id in beanstalk queue; beanstalk doesn't accept queue ids, so we must
  // use theirs to snoop job state
  // Can't make this required, because introduces a race condition;
  // Need to submit to the queue, before we save, so if worker responds too quickly
  // We cannot find the job using the queueID
  queueID: {
    type: String
  },
  submittedDate: Date,
  startedDate: Date,
  //This is completed or failed
  finishedDate: Date,
  // we don't have meaningful defaults for these; set by the caller
  queueStats: {
    ttr: Number,
    age: Number
  },
  type: {
    type: String,
    required: true
  },

  // Index submission schema just like regular queueSchema, but allows added files
  // User can submit a bed file for example
  // We expect this to be enough to fully locate the file;
  // Because dirs.in in the parent schema defiens the directory
  // file path
  // This is completely optional, not every user of this schema will have this
  addedFileNames: [String]
};

// TODO: allow instance* methods to accept an object
// Previous attempt at this didn't work..
module.exports.instanceJobQueue = function() {
  return new Schema(base);
};

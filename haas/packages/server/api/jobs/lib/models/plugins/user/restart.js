const fs = require('fs-extra');

exports = module.exports = function restarPlugin(schema) {
  schema.methods.restartJob = function createUserJob(cb) {
    const self = this;
    if (!this.inputFilePath) {
      return cb(
        new Error(`This job doesn't have an inputFilePath. Is the job new?`)
      );
    }
    fs.stat(this.inputFilePath, function statCb(exists) {
      if (!exists) {
        return cb(
          new Error(`This job's inputFilePath points to an unreachable file!`)
        );
      }
      self.wasNew = true;
      self.save(function(err) {
        if (err) {
          return cb(err);
        }
        return cb(null, self);
      });
    });
  };
};

const createPlugin = require('./create');
// const restartPlugin = require('./restart');

exports.set = function setUserPlugins(Schema, options) {
  Schema.plugin(createPlugin, options);
  // Schema.plugin(restartPlugin, options);
  return Schema;
};

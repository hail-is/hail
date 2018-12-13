const mongoose = require.main.require('./common/database');
const UserSchema = require('./schemas/user');
const resetPlugin = require('./plugins/reset');
const rolesPlugin = require('./plugins/roles');
const publicPlugin = require('./plugins/public');

let modelInstance;
module.exports = function uMfactory(config) {
  if (modelInstance) {
    return modelInstance;
  }

  if (!(config) ) {
    throw new Error('Need user and jobComm instance');
  }

  const uSchema = UserSchema(config);

  resetPlugin.set(uSchema);
  rolesPlugin.set(uSchema, config);
  publicPlugin.set(uSchema);

  modelInstance = mongoose.model('User', uSchema);

  return modelInstance;
};

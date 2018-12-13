const UserModels = require('./lib/models');
const UserRouter = require('./lib/router');
const Auth = require('./lib/auth');

let uF;
module.exports = function userFactory(app, config) {
  if (uF) {
    return uF;
  }

  uF = {
    routesRootName: 'user'
  };

  uF.config = config.user;
  uF.Model = UserModels(uF.config);
  uF.tokenManager = new Auth.TokenManager(uF.Model, uF.config);
  uF.middleware = new Auth.AuthMiddleware(uF.Model, uF.tokenManager);
  uF.router = new UserRouter(app, uF).router;

  uF.getUser = function getUser(request, cb) {
    const user = request[uF.tokenManager.attachProperty] || request;
    let err;
    if (!Object.keys(user)) {
      err = new Error('No user found');
    }
    if (cb) {
      return cb(err, user);
    }
    return user;
  };

  uF.isGuest = function isGuest(user) {
    return uF.Model.isGuest(user);
  };

  // TODO: Move to just ._id
  uF.getUserId = function getUserId(user) {
    // sub is what auth0 uses
    return user && (user.id || user._id || user.sub);
  };

  return uF;
};

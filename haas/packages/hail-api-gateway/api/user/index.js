const Auth = require('./lib/auth');

let uF;
module.exports = function userFactory(config = {}) {
  if (uF) {
    return uF;
  }

  uF = {
    routesRootName: 'user'
  };

  uF.middleware = new Auth.AuthMiddleware();

  return uF;
};

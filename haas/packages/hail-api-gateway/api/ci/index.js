const Router = require('./lib/router');

let aF;
exports = module.exports = function jobsFactory(User) {
  if (aF) {
    return aF;
  }

  aF = {};

  const router = Router(User);
  //Register our router with the application router
  aF.routes = router.routes;

  return aF;
};

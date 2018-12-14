const Router = require('./lib/router');

let aF;
exports = module.exports = function jobsFactory(User, config) {
  if (aF) {
    return aF;
  }

  aF = {
    routesRootName: 'ci'
  };

  const awsConfig = config.ci;
  const router = new Router(User, awsConfig);

  //Register our router with the application router
  aF.router = router.router;

  return aF;
};

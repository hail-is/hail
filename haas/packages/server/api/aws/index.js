const AWSrouter = require('./lib/router');

let aF;
exports = module.exports = function jobsFactory(User, config) {
  if (aF) {
    return aF;
  }

  aF = {
    routesRootName: 'aws',
  };

  const awsConfig = config.aws;
  const awsRouter = new AWSrouter(User, awsConfig);

  //Register our router with the application router
  aF.router = awsRouter.router;

  return aF;
};
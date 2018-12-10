const Client = require('./lib/client');
// const Server = require('./lib/server');

let cF;
exports = module.exports = function commFactory(httpServer, user, config) {
  if (cF) {
    return cF;
  }
  cF = {};

  const clientConfig = config.comm;
  cF.client = new Client(httpServer, user, clientConfig);

  return cF;
};

// const ClientComm = require('./lib/clientComm');
// const ServerComm = require('./lib/server');
// class Communication {
//   constructor(appServer, auth, opts) {
//     const myOpts = opts || {}; // no defaults in v8 yet
//     this.client = new ClientComm(appServer, auth, myOpts.client);
//     this.Server = ServerComm;
//   }
// }
// // renforce singleton
// let instance;
// module.exports.instance = function commmFactory(appServer, auth, opts) {
//   if (instance instanceof Communication) {
//     return instance;
//   }
//   instance = new Communication(appServer, auth, opts);

//   return instance;
//  };

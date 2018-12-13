// ensure permissions; could also require user to do in shell, but seems
// unlikely that they would do that.
// mask '002' will prevent u+w
// default umask is '022' which prevents u+w and g+w
process.umask('002');
// ==================================
module.exports = {
  // MongoDB connection options
  server: {
    allowedCORSOrigins: 'all',
  },
  database: {
    seedDB: true,
  },
  comm: {
    client: {
      serveClient: true,
      transports: ['websocket', 'polling', 'xhr-polling', 'jsonp-polling'],
    },
  },
  jobs: {
    maxJobs: 1,
  },
};

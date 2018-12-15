// ensure permissions; could also require user to do in shell, but seems
// unlikely that they would do that.
// mask '002' will prevent u+w
// default umask is '022' which prevents u+w and g+w
// ==================================
module.exports = {
  // MongoDB connection options
  server: {
    allowedCORSOrigins: 'all',
  },
  database: {
    username: '',
    password: '',
    options: {
      db: {
        safe: true,
      },
    },
    seedDB: true,
  },
  user: {
    auth: {
      token: {
        expiration: 60 * 24 * 7 * 2, // in minutes, 2 weeks
        refreshExpiration: 60 * 24 * 7 * 4, // in minutes, 4 weeks
      }
    },
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

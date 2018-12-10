const path = require('path');
const _ = require('lodash');

if (!process.env.NODE_ENV) {
  throw new Error('You must set the NODE_ENV environment variable');
}

// All configurations will extend these options
// ============================================
const all = {
  env: process.env.NODE_ENV,

  server: {
    root: path.normalize(path.resolve(__dirname, '/../../..') ),
    port: process.env.PORT || 9001,
    cpuMax: process.env.CPU_MAX || 4,
  },
  database: {
    uri: process.env.MONGODB_URI,
    database: process.env.MONGODB_DATABASE,
    user: process.env.MONGODB_USER,
    password: process.env.MONGODB_PASSWORD,
    options: process.env.MONDODB_OPTIONS,
  },
  router: {
    publicPath: path.resolve('../public/'),
  },
  comm: {
    server: {
      host: process.env.COMM_SERVER_HOST,
      port: process.env.COMM_SERVER_PORT,
    },
  },
  jobs: {
    schema: {
      baseDir: process.env.JOBS_DIR,
      jobCollection: process.env.MONDODB_JOB_COLLECTION,
    },
    comm: {
      server: {
        host: process.env.COMM_SERVER_HOST,
        port: process.env.COMM_SERVER_PORT,
      },
    },
    download: {
      secret: process.env.DOWNLOAD_SECRET,
    },
    queues: {
      configPath: process.env.QUEUE_CONFIG_PATH,
    },
    configDir: process.env.JOB_CONFIG_DIR,
    elastic: {
      hosts: process.env.ELASTIC_HOSTS.split(','),
      timeout: process.env.ELASTIC_TIMEOUT || 6.0e5,
      requestTimeout: process.env.ELASTIC_REQUEST_TIMEOUT || 6.0e5,
      apiVersion: process.env.ELASTIC_API_VERSION,
      sniffOnStart: process.env.ELASTIC_SNIFF_ON_START || true,
      snpiffInterval: process.env.SNIFF_INTERVAL || 60000,
      //Can't use sniffing, doesn't keep alive for some reason
      // keepAlive: true,
      // sniffOnStart: true,
      // sniffInterval: 3.6e6, //milliseconds ; about 1 hour
      // sniffOnConnectionFault: true,
      // suggestCompression: true,
    }
  },
  user: {
    // lowest level role is always the one with least priveleges
    // lowest level role doesn't persist
    roles: ['guest', 'user', 'admin'],
    auth: {
      strategies: {
        guest: {},
        local: {},
        // facebook: {
        //   clientID: process.env.FACEBOOK_ID || 'id',
        //   clientSecret: process.env.FACEBOOK_SECRET || 'secret',
        //   callbackURL: (process.env.DOMAIN || '') + '/auth/facebook/callback',
        // },

        // twitter: {
        //   clientID: process.env.TWITTER_ID || 'id',
        //   clientSecret: process.env.TWITTER_SECRET || 'secret',
        //   callbackURL: (process.env.DOMAIN || '') + '/auth/twitter/callback',
        // },

        // google: {
        //   clientID: process.env.GOOGLE_ID || 'id',
        //   clientSecret: process.env.GOOGLE_SECRET || 'secret',
        //   callbackURL: (process.env.DOMAIN || '') + '/auth/google/callback',
        // },
      },
      token: {
        attachProperty: 'user',
        expiration: process.env.AUTH_TOKEN_EXPIRATION,
        refreshExpiration: process.env.AUTH_REFRESH_TOKEN_EXPIRATION,
        secret: process.env.AUTH_SECRET,
        refreshSecret: process.env.AUTH_REFRESH_SECRET,
      }
    },
  },
};

// Export the config object based on the NODE_ENV
// ==============================================
module.exports = _.merge(
  all, require('./environment/' + process.env.NODE_ENV + '.js') || {}
);

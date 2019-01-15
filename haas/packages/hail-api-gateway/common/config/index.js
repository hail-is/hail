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
    root: path.normalize(path.resolve(__dirname, '/../../..')),
    port: process.env.PORT || 8000,
    cpuMax: process.env.CPU_MAX || 2
  }
};

// Export the config object based on the NODE_ENV
// ==============================================
module.exports = _.merge(
  all,
  require('./environment/' + process.env.NODE_ENV + '.js') || {}
);

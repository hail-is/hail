//populate process.env
require('dotenv').load();

require('dotenv').config('./.env');
const InvalidTokenError = require.main.require(
  './common/auth/errors/InvalidTokenError'
);
const polka = require('polka');
const fs = require('fs-extra'); // adds functions like mkdirp (mkdir -p)
const http = require('http');
const { json } = require('body-parser');
const { makeExecutableSchema } = require('graphql-tools');
const { ApolloServer } = require('apollo-server-express');
const { mergeTypes } = require('merge-graphql-schemas');
const _ = require('lodash');

// local lib modules
const log = require('./common/logger');
const config = require('./common/config');

const { PORT = 8000 } = process.env;

// graphql

const CI = require('./api/ci');
const userFactory = require('./api/user');
const { jobSchema, jobResolver } = require('./api/jobs');

const user = userFactory(config);
// socketio requires the http.createServer return obj not app
// const comm = Comm(httpServer, user, config);
// const jobs = Jobs(comm, user, config);
// const aws = SeqAws(user, config);
const ci = CI(user, config);

// const jobResolver = JobResolver(jobs.jobModel, user.middleware.getUserId);
const schemas = mergeTypes([jobSchema]);
const resolvers = _.merge([jobResolver]);

/**
 * Express configuration.
 */
fs.ensureDir(config.router.publicPath, err => {
  if (err) {
    throw new Error(err);
  }
});

fs.ensureDir('./logs', err => {
  if (err) {
    throw new Error(err);
  }
});

const httpServer = http.createServer();
const app = polka({
  server: httpServer
});

app.use('/', (req, res, next) => {
  res.setHeader('X-Frame-Options', 'SAMEORIGIN');
  res.setHeader('Access-Control-Allow-Credentials', true);
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET,POST');
  //X-Requested-With is for CSRF
  //https://stackoverflow.com/questions/17478731/whats-the-point-of-the-x-requested-with-header/22533680
  res.setHeader(
    'Access-Control-Allow-Headers',
    'Authorization, X-Requested-With, X-HTTP-Method-Override, Content-Type, Accept, Cache-Control'
  );

  if (req.method === 'OPTIONS') {
    res.end();
  } else {
    next();
  }
});

// const schema = (module.exports = makeExecutableSchema({ typeDefs, resolvers }));

app.use('/graphql', user.middleware.verifyToken);

const apolloServer = new ApolloServer({
  typeDefs: schemas,
  resolvers,
  context: ({ req }) => {
    // Look at the request to run custom user logic
    return { user: req.user };
  },
  engine: {
    apiKey: process.env.APOLLO_ENGINE_API_KEY
  }
});

const routes = [...ci.routes];

routes.forEach(route => {
  if (route.method === 'GET' || route.method === 'get') {
    // spread, middleware and terminating route handle
    app.get(route.path, ...route.handlers);
  }
});

app.listen(PORT, err => {
  if (err) throw err;
  console.log(`Ready on port ${PORT}`);
});

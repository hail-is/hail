/**
 * Module dependencies.
 */

/* TODO:
 * Consume User model
 * Ensure no DDOS, other security
 * Clean up CORS handling including OPTIONS requests
 * Implement socketIO clustering, investigate redis clustering
 * Check what happens if move from tempfolder
 * during upload fails (does session get stored if res.status(500).end())
 * Todo, build in auth for registered userSessions
 * Delete all files (done)
 * Add user name support
 * Improve error handling, give client feedback
 * Either remove user uploaded files once job completes,
 * cron to periodically delete,
 * or keep (speak to Viren about storage requirements)
 */
// TODO implement clustering and take a look at socket.io-emitter
// https://github.com/Automattic/socket.io-emitter

// fill process.env
require('dotenv').load();
process.env.NODE_TLS_REJECT_UNAUTHORIZED = 0;
/* Express related*/
const express = require('express');
const fs = require('fs-extra'); // adds functions like mkdirp (mkdir -p)
const bodyParser = require('body-parser');
const cookieParser = require('cookie-parser');
const compression = require('compression');
const morgan = require('morgan');
const http = require('http');
const helmet = require('helmet');
const _ = require('lodash');
const cors = require('cors');
const { ApolloServer } = require('apollo-server-express');
const { mergeTypes } = require('merge-graphql-schemas');

// local lib modules
const log = require('./common/logger');
const config = require('./common/config');

// graphql
const { postSchema, postResolver } = require('./api/post');
const { jobSchema, JobResolver } = require('./api/jobs');
const { s3Schema, s3Resolver } = require('./api/s3');

// REST and models
const User = require('./api/user');
const Comm = require('./api/communication');
const { Jobs } = require('./api/jobs');
const SeqAws = require('./api/aws');
const CI = require('./api/ci');
const appRouter = require('./router');

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

// const filter = require('content-filter')
// const mongoSanitize = require('express-mongo-sanitize');

/**
 * Create Express server.
 */
const corsOptions = {
  origin: 'localhost:3000',
  optionsSuccessStatus: 200 // some legacy browsers (IE11, various SmartTVs) choke on 204
};

const app = express(); //cors(corsOptions)

// require('pmx').init({
//   http: true, // HTTP routes logging (default: true)
//   ignore_routes: [/socket\.io/, /notFound/], // Ignore http routes with this pattern (Default: [])
//   errors: true, // Exceptions loggin (default: true)
//   custom_probes: true, // Auto expose JS Loop Latency and HTTP req/s as custom metrics
//   network: true, // Network monitoring at the application level
//   ports: true,  // Shows which ports your app is listening on (default: false)
//   alert_enabled: true, // Enable alert sub field in custom metrics   (default: false)
// });

// app.use(helmet());
app.use(express.static(config.router.publicPath));

app.set('port', 8000); //config.server.port);
app.use(compression());
app.use(
  bodyParser.json({
    limit: '1000mb'
  })
);
app.use(
  bodyParser.urlencoded({
    limit: '1000mb',
    extended: true
  })
);

// function allowCrossDomain(req, res, next) {
//   res.header('Access-Control-Allow-Origin *'); // config.server.allowedCORSOrigins
//   res.header('Access-Control-Allow-Credentials', true);
//   res.header('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS');
//   res.header(
//     'Access-Control-Allow-Headers',
//     'Content-Type, Authorization, Content-Length, X-Requested-With'
//   );

//   // intercept OPTIONS method
//   if (req.method === 'OPTIONS') {
//     res.status(200).end();
//   } else {
//     next();
//   }
// }

// app.use(allowCrossDomain);

app.use((req, res, next) => {
  res.header('X-Frame-Options', 'SAMEORIGIN');
  res.header('Access-Control-Allow-Credentials', true);
  res.header('Access-Control-Allow-Origin', '*');
  res.header('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE');
  res.header(
    'Access-Control-Allow-Headers',
    'Authorization, X-Requested-With, X-HTTP-Method-Override, Content-Type, Accept, Cache-Control'
  );
  if (req.method === 'OPTIONS') {
    // console.info('Option');
    res.status(200).end();
  } else {
    next();
  }
});

//Will replace keys containing ".", "$" with _
// app.use(mongoSanitize({
//   replaceWith: '_',
// }));

//Sanitize user inputs
//This breaks a LOT (no saving/updating User or Job models)
// app.use(filter());

app.use(cookieParser());

// // initialize everything shared

if (process.env.NODE_ENV === 'development') {
  fs.ensureDir('./logs/heapdumps');
}

// Grab our .env configuration (show path to .env config explicitly)
require('dotenv').config('./.env');

const httpServer = http.createServer(app);

// TODO: restrict this for upload routes
// Don't close connections, even if they've been held open a long time
httpServer.timeout = 0;

// Configure REST endpoints, models

const user = User(app, config);
// socketio requires the http.createServer return obj not app
const comm = Comm(httpServer, user, config);
const jobs = Jobs(comm, user, config);
const aws = SeqAws(user, config);
const ci = CI(user, config);

const jobResolver = JobResolver(jobs.jobModel, user.getUserId);
const schemas = mergeTypes([postSchema, jobSchema, s3Schema]);
const resolvers = _.merge(postResolver, jobResolver, s3Resolver);

app.use(
  morgan('combined', {
    stream: log.stream
  })
);

appRouter(app, [user, jobs, aws, ci], config);

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

apolloServer.applyMiddleware({ app, path: '/graphql' });

apolloServer.installSubscriptionHandlers(httpServer);

const port = app.get('port');
httpServer.listen(port, () => {
  console.log(
    `🚀 Server ready at http://localhost:${port}${apolloServer.graphqlPath}`
  );
  console.log(
    `🚀 Subscriptions ready at ws://localhost:${port}${
      apolloServer.subscriptionsPath
    }`
  );
});

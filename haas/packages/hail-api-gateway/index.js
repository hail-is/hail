//populate process.env
require('dotenv').load();
require('dotenv').config('./.env');

const polka = require('polka');
const fs = require('fs-extra'); // adds functions like mkdirp (mkdir -p)
const http = require('http');
// TODO: If no desire for GraphQL expressed, remove
// const { ApolloServer } = require('apollo-server-express');
// const { mergeSchemas, makeExecutableSchema } = require('graphql-tools');

// local lib modules
const config = require('./common/config');

const { PORT = 8000 } = process.env;

// graphql

const CI = require('./api/ci');
const userFactory = require('./api/user');
// const github = require('./api/github');

const user = userFactory(config);

const ci = CI(user, config);

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
  res.setHeader('X-Content-Type-Options', 'nosniff');
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

const routes = [...ci.routes];

routes.forEach(route => {
  if (route.method === 'GET' || route.method === 'get') {
    // spread, middleware and terminating route handle
    app.get(route.path, ...route.handlers);
  }
});

// Example w/ graphQL
// TODO: If we decide no graphql endpoints needed, remove
(async () => {
  // Await remote schema
  //   const githubSchema = await github(user);

  //   const schema = mergeSchemas({
  //     schemas: [
  //       makeExecutableSchema({ typeDefs: jobSchema, resolvers: jobResolver }),
  //       githubSchema
  //     ]
  //   });

  //   const apolloServer = new ApolloServer({
  //     // typeDefs: jobSchema,
  //     // resolvers: jobResolver,
  //     schema,
  //     // resolvers,
  //     context: ({ req }) => {
  //       // Look at the request to run custom user logic
  //       return { user: req.user };
  //     },
  //     engine: {
  //       apiKey: process.env.APOLLO_ENGINE_API_KEY
  //     },
  //     playground: {
  //       endpoint: '/graphql'
  //       // subscriptionEndpoint?: string
  //     },
  //     cacheControl: {
  //       defaultMaxAge: 5,
  //       stripFormattedExtensions: false,
  //       calculateCacheControlHeaders: false
  //     }
  //   });

  //   app.use('/graphql', user.middleware.verifyToken);
  //   apolloServer.applyMiddleware({ app }, '/graphql');

  app.listen(PORT, err => {
    if (err) throw err;
    console.log(`Ready on port ${PORT}`);
  });
})();

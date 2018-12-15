'use strict';

const helmet = {
  contentSecurityPolicy: require('helmet-csp'),
  dnsPrefetchControl: require('dns-prefetch-control'),
  expectCt: require('expect-ct'),
  featurePolicy: require('feature-policy'),
  frameguard: require('frameguard'),
  hidePoweredBy: require('hide-powered-by'),
  hsts: require('hsts'),
  ieNoOpen: require('ienoopen'),
  noCache: require('nocache'),
  noSniff: require('dont-sniff-mimetype'),
  permittedCrossDomainPolicies: require('helmet-crossdomain'),
  referrerPolicy: require('referrer-policy'),
  xssFilter: require('x-xss-protection')
};
const config = require('./fastify-helmet-config.js');

const middlewares = Object.keys(helmet);

module.exports = () =>
  middlewares.map(middlewareName => {
    const middleware = helmet[middlewareName];
    const option = opts[middlewareName];
    const isDefault = config.defaultMiddleware.indexOf(middlewareName) !== -1;

    if (option === false) {
      return;
    }

    if (option != null) {
      if (option === true) {
        return middleware({});
      } else {
        return middleware(option);
      }
    } else if (isDefault) {
      return middleware({});
    }

    return;
  });

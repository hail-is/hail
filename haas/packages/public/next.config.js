const withCSS = require('@zeit/next-css');
const withSass = require('@zeit/next-sass');
const withTypescript = require('@zeit/next-typescript');
const withPurgeCss = require('next-purgecss');
const ForkTsCheckerWebpackPlugin = require('fork-ts-checker-webpack-plugin');
const withOffline = require('next-offline');

require('dotenv').config('.env');

module.exports = withOffline(
  withTypescript(
    withSass(
      withCSS(
        withPurgeCss({
          purgeCss: {
            keyframes: true, //for animate.css
            fontFace: true
          },
          workboxOpts: {},
          webpack(config, options) {
            if (options.isServer) {
              config.plugins.push(new ForkTsCheckerWebpackPlugin());
            }

            return config;
          }
        })
      )
    )
  )
);

const serverRuntimeConfig = {
  // Will only be available on the server side
  GITHUB: {
    ACCESS_TOKEN: process.env.GITHUB_ACCESS_TOKEN
  }
};

const publicRuntimeConfig = {
  // Will be available on both server and client
  staticFolder: '/static',
  API_UPLOAD_URL:
    process.env.API_UPLOAD_URL || 'http://localhost:8000/api/jobs/upload',
  API_DOWNLOAD_URL:
    process.env.API_DOWNLOAD_URL || 'http://localhost:8000/api/jobs/download',
  GRAPHQL: {
    ENDPOINT: process.env.GRAPHQL_ENDPOINT || 'http://localhost:8000/graphql'
  },
  AUTH0: {
    DOMAIN: process.env.AUTH0_DOMAIN || 'hail.auth0.com',
    AUDIENCE: process.env.AUTH0_AUDIENCE,
    REDIRECT_URI:
      process.env.AUTH0_REDIRECT_URI || 'http://localhost:3000/auth0callback',
    CLIENT_ID: process.env.AUTH0_CLIENT_ID,
    RESPONSE_TYPE: process.env.AUTH0_RESPONSE_TYPE || 'token id_token',
    SCOPE: 'openid profile' //process.env.AUTH0_SCOPE ||
  },
  GITHUB: {
    ACCESS_TOKEN_UNSAFE: process.env.GITHUB_ACCESS_TOKEN
  },
  SCORECARD: { URL: process.env.SCORECARD_URL || 'http://localhost:5000/json' }
};

serverRuntimeConfig.SCORECARD = publicRuntimeConfig.SCORECARD;

module.exports.publicRuntimeConfig = publicRuntimeConfig;
module.exports.serverRuntimeConfig = serverRuntimeConfig;

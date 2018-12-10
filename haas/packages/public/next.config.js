const withCSS = require('@zeit/next-css');
const withSass = require('@zeit/next-sass');
const withTypescript = require('@zeit/next-typescript');
const withPurgeCss = require('next-purgecss');
const ForkTsCheckerWebpackPlugin = require('fork-ts-checker-webpack-plugin');
require('dotenv').config('.env');

module.exports = withTypescript(
  withSass(
    withCSS(
      withPurgeCss({
        purgeCss: {
          keyframes: true, //for animate.css
          fontFace: true
        },
        webpack(config, options) {
          if (options.isServer) {
            config.plugins.push(new ForkTsCheckerWebpackPlugin());
          }
          return config;
        }
      })
    )
  )
);

module.exports.serverRuntimeConfig = {
  // Will only be available on the server side
  GITHUB: {
    ACCESS_TOKEN: process.env.GITHUB_ACCESS_TOKEN
  }
};

module.exports.publicRuntimeConfig = {
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
    CLIENT_ID:
      process.env.AUTH0_CLIENT_ID || 'PYHMcItGAEL2PCvBJCEF05tFsdMkt4GG',
    RESPONSE_TYPE: process.env.AUTH0_RESPONSE_TYPE || 'token id_token',
    SCOPE: 'openid profile repo read:user delete_repo admin:repo_hook' //process.env.AUTH0_SCOPE ||
  }
};

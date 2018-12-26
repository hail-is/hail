const withCSS = require('@zeit/next-css');
const withSass = require('@zeit/next-sass');
const withTypescript = require('@zeit/next-typescript');
// const withPurgeCss = require('next-purgecss');
const ForkTsCheckerWebpackPlugin = require('fork-ts-checker-webpack-plugin');
const MonacoWebpackPlugin = require('monaco-editor-webpack-plugin');

require('dotenv').config('.env');

module.exports = withTypescript(
  withSass(
    withCSS({
      webpack(config, options) {
        config.resolve.modules.push('./');

        if (options.isServer) {
          config.plugins.push(new ForkTsCheckerWebpackPlugin());
        } else {
          // browser only
          config.plugins.push(
            new MonacoWebpackPlugin({
              output: 'static/',
              languages: [
                'javascript',
                'typescript',
                'python',
                'r',
                'perl',
                'shell'
              ]
            })
          );

          if (config.optimization.splitChunks.cacheGroups.commons) {
            config.optimization.splitChunks.cacheGroups.commons.minChunks = 2;
          }
        }

        return config;
      }
    })
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
    process.env.API_UPLOAD_URL || 'https://api.localhost/jobs/upload',
  API_DOWNLOAD_URL:
    process.env.API_DOWNLOAD_URL || 'https://api.localhost/jobs/download',
  GRAPHQL: {
    ENDPOINT: process.env.GRAPHQL_ENDPOINT || 'https://api.localhost/graphql'
  },
  AUTH0: {
    DOMAIN: process.env.AUTH0_DOMAIN || 'hail.auth0.com',
    AUDIENCE: process.env.AUTH0_AUDIENCE,
    REDIRECT_URI:
      process.env.AUTH0_REDIRECT_URI || 'https://localhost/auth0callback',
    CLIENT_ID: process.env.AUTH0_CLIENT_ID,
    RESPONSE_TYPE: process.env.AUTH0_RESPONSE_TYPE || 'token id_token',
    SCOPE: 'openid profile' //process.env.AUTH0_SCOPE ||
  },
  GITHUB: {
    ACCESS_TOKEN_UNSAFE: process.env.GITHUB_ACCESS_TOKEN
  },
  SCORECARD: {
    URL: process.env.SCORECARD_URL || 'https://scorecard.localhost/json',
    USER_URL:
      process.env.SCORECARD_USER_URL || 'https://scorecard.localhost/json/users'
  },
  CI: {
    ROOT_URL: process.env.CI_ROOT_URL || 'https://api.localhost/ci'
  }
};

serverRuntimeConfig.SCORECARD = publicRuntimeConfig.SCORECARD;

module.exports.publicRuntimeConfig = publicRuntimeConfig;
module.exports.serverRuntimeConfig = serverRuntimeConfig;

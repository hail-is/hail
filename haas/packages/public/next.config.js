const withCSS = require('@zeit/next-css');
const withSass = require('@zeit/next-sass');
const withTypescript = require('@zeit/next-typescript');
const withPurgeCss = require('next-purgecss');
const ForkTsCheckerWebpackPlugin = require('fork-ts-checker-webpack-plugin');
const glob = require('glob');
const path = require('path');
require('dotenv').config('.env');

module.exports = withTypescript(
  // Purge not currently working with monaco
  withPurgeCss(
    withSass(
      withCSS({
        purgeCssOptions: {
          fontFace: true,
          keyframes: true
        },
        webpack(config, options) {
          config.resolve.modules.push('./');

          if (options.isServer) {
            config.plugins.push(new ForkTsCheckerWebpackPlugin());
          } else {
            if (config.optimization.splitChunks.cacheGroups.commons) {
              config.optimization.splitChunks.cacheGroups.commons.minChunks = 2;
            }
          }

          // Adapted from https://github.com/rohanray/next-fonts/blob/master/index.js
          const testPattern = /\.(woff|woff2|eot|ttf|otf|svg)$/;

          config.module.rules.push({
            test: testPattern,
            use: [
              {
                loader: 'url-loader',
                options: {
                  limit: 8192,
                  fallback: 'file-loader',
                  publicPath: `/_next/static/fonts/`,
                  outputPath: 'static/fonts/',
                  name: '[name]-[hash].[ext]'
                }
              }
            ]
          });

          return config;
        }
      })
    )
  )
);

const serverRuntimeConfig = {
  // Will only be available on the server side
};

const publicRuntimeConfig = {
  // Will be available on both server and client
  staticFolder: '/static',
  GRAPHQL: {
    ENDPOINT: process.env.GRAPHQL_ENDPOINT || 'https://api.localhost/graphql'
  },
  AUTH0: {
    DOMAIN: process.env.AUTH0_DOMAIN || 'hail.auth0.com',
    AUDIENCE: process.env.AUTH0_AUDIENCE,
    REDIRECT_URI: process.env.AUTH0_REDIRECT_URI,
    CLIENT_ID: process.env.AUTH0_CLIENT_ID,
    RESPONSE_TYPE: process.env.AUTH0_RESPONSE_TYPE || 'token id_token',
    SCOPE: 'openid profile' //process.env.AUTH0_SCOPE ||
  },
  SCORECARD: {
    URL: process.env.SCORECARD_URL || 'https://scorecard.localhost/json',
    USER_URL:
      process.env.SCORECARD_USER_URL || 'https://scorecard.localhost/json/users'
  },
  NOTEBOOK: {
    URL: process.env.CI_ROOT_URL || 'https://api.localhost/notebook'
  }
};

serverRuntimeConfig.SCORECARD = publicRuntimeConfig.SCORECARD;

module.exports.publicRuntimeConfig = publicRuntimeConfig;
module.exports.serverRuntimeConfig = serverRuntimeConfig;

require('dotenv').config();

const withTypescript = require('@zeit/next-typescript');
const ForkTsCheckerWebpackPlugin = require('fork-ts-checker-webpack-plugin');
const withPurgeCss = require('next-purgecss');
const withCss = require('@zeit/next-css');
const withSass = require('@zeit/next-sass');
const withPlugins = require('next-compose-plugins');

const publicRuntimeConfig = {
  AUTH0: {
    DOMAIN: process.env.AUTH0_DOMAIN,
    SCOPE: process.env.AUTH0_SCOPE,
    CALLBACK_SUFFIX: process.env.AUTH0_CALLBACK_SUFFIX,
    RESPONSE_TYPE: process.env.AUTH0_RESPONSE_TYPE,
    CLIENT_ID: process.env.AUTH0_CLIENT_ID,
    AUDIENCE: process.env.AUTH0_AUDIENCE
  },
  SCORECARD: {
    DOMAIN: process.env.SCORECARD_DOMAIN,
    USERS: process.env.SCORECARD_USERS
      ? process.env.SCORECARD_USERS.split(',')
      : []
  },
  NOTEBOOK: {
    DOMAIN: process.env.NOTEBOOK_DOMAIN
  }
};

const nextConfig = {
  distDir: 'build',
  webpack: (config, options) => {
    if (options.isServer) {
      config.plugins.push(new ForkTsCheckerWebpackPlugin());
    } else if (!options.dev) {
      // Only in a production environment, in the client-side webpack phase
      // does next configure splitChuunks
      // In practice this happens during "next build", and we don't care about
      // optimizing chunk settings during dev
      config.optimization.splitChunks.cacheGroups.commons.minChunks = 2;
    }

    return config;
  },
  publicRuntimeConfig
};

module.exports = withPlugins(
  [
    withTypescript,
    withSass,
    withCss,
    [
      withPurgeCss,
      {
        // regular purgeCss options
        // https://www.purgecss.com
        purgeCss: {
          keyrframes: true,
          fontFace: true
        },
        // Plugin specific: Specifiy which files should be checked
        // before deciding some imported css is not used:
        purgeCssPaths: ['pages/**/*', 'components/**/*']
      }
    ]
  ],
  nextConfig
);

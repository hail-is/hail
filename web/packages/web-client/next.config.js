const withTypescript = require('@zeit/next-typescript');
const ForkTsCheckerWebpackPlugin = require('fork-ts-checker-webpack-plugin');
const withPurgeCss = require('next-purgecss');
const withCss = require('@zeit/next-css');
const withSass = require('@zeit/next-sass');
const withPlugins = require('next-compose-plugins');

// If above plugins need to be run only during server-build phase
// const {PHASE_DEVELOPMENT_SERVER} = require('next/constants')

const nextConfig = {
  distDir: 'build',
  webpack: (config, options) => {
    if (options.isServer) {
      config.plugins.push(new ForkTsCheckerWebpackPlugin());
    }

    return config;
  }
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

module.exports.publicRuntimeConfig = {
  AUTH0: {
    scope: '',
    domain: ''
  }
};

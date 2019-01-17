const withTypescript = require('@zeit/next-typescript');
const ForkTsCheckerWebpackPlugin = require('fork-ts-checker-webpack-plugin');
const withPurgeCss = require('next-purgecss');
const withCss = require('@zeit/next-css');
const withSass = require('@zeit/next-sass');

module.exports = withPurgeCss(
  withSass(
    withCss(
      withTypescript({
        purgeCssOptions: {
          fontFamily: true
        },
        webpack(config, options) {
          if (options.isServer) {
            config.plugins.push(new ForkTsCheckerWebpackPlugin());
          } else {
            // https://github.com/zeit/next.js/issues/5923
            // config.optimization.splitChunks.cacheGroups.default = {
            //   minChunks: 2,
            //   reuseExistingChunk: true
            // };
            // config.optimization.splitChunks.minChunks = 2;
          }

          return config;
        }
      })
    )
  )
);

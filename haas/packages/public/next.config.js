const withCSS = require('@zeit/next-css');
const withSass = require('@zeit/next-sass');
const withTypescript = require('@zeit/next-typescript');
const withPurgeCss = require('next-purgecss');
const glob = require('glob');
const path = require('path');
const ForkTsCheckerWebpackPlugin = require('fork-ts-checker-webpack-plugin');
const MonacoWebpackPlugin = require('monaco-editor-webpack-plugin');
var fs = require('fs');

require('dotenv').config('.env');

// Return a list of files of the specified fileTypes in the provided dir,
// with the file path relative to the given dir
// dir: path of the directory you want to search the files for
// fileTypes: array of file types you are search files, ex: ['.txt', '.jpg']
function getFilesFromDir(dir, fileTypes) {
  const filesToReturn = [];
  const re = new RegExp('(' + fileTypes.join('|') + ')' + '$');

  function walkDir(currentPath) {
    const files = fs.readdirSync(currentPath);
    for (const file of files) {
      const cPath = path.join(currentPath, file);
      // console.info(cPath, typeof cPath);

      if (fs.existsSync(cPath) && fs.lstatSync(cPath).isDirectory()) {
        walkDir(cPath);
      }

      if (re.test(cPath)) {
        filesToReturn.push(cPath);
      }
    }
  }
  walkDir(path.join(__dirname, dir));
  return filesToReturn;
}

//print the txt files in the current directory
const paths = [].concat(
  getFilesFromDir('./', ['.js', '.jsx', '.ts', '.tsx', '.css', '.scss'])
  // getFilesFromDir('pages', ['.js', '.jsx', '.ts', '.tsx']),
  // getFilesFromDir('components', ['.js', '.jsx', '.ts', '.tsx']),
  // getFilesFromDir('node_modules/monaco-editor', ['.js', '.jsx', '.ts', '.tsx'])
);

module.exports = withTypescript(
  // Purge not currently working with monaco
  // withPurgeCss(
  withSass(
    withCSS({
      // purgeCssOptions: {
      //   fontFace: true,
      //   keyframes: true,
      //   // paths: paths,
      //   whitelistPatternsChildren: /monaco-editor$/,
      //   rejected: true
      // },
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
                'json',
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
  // )
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

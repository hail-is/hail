const express = require('express');
const passport = require('passport');
const fs = require('fs');
const path = require('path');

const skipMethods = ['google', 'facebook', 'twitter'];
class AuthStrategies {
  constructor(app, user) {
    app.use(passport.initialize() );
    const tokenManager = user.tokenManager;
    const middleware = user.middleware;
    const UserModel = user.Model;
    const config = user.config.auth.strategies || {};
    
    const router = express.Router({mergeParams: true});

    const strats = _getDirectories();

    // TODO: change to use the full strat object, passing the config.
    strats.forEach(function(stratName) {
      if (skipMethods.indexOf(stratName) > -1) { return; }
     
      const stratPath = './' + stratName;

      const stratConfig = config[stratName] || {};

      require(stratPath + '/passport')
      .setup(UserModel, stratConfig, tokenManager);

      const stratRouter = require(stratPath);
      const srInstance = stratRouter(tokenManager, middleware);
     
      router.use('/' + stratName, srInstance);
    });

    this.router = router;
    // console.info("another router stack", this.router.stack);
  }
}

module.exports = AuthStrategies;

// http://stackoverflow.com/questions/18112204/get-all-directories-within-directory-nodejs

function _getDirectories() {
  return fs.readdirSync(__dirname).filter(function(file) {
    return fs.statSync(path.resolve(__dirname, file)).isDirectory();
  });
}


const passport = require('passport');
const TwitterStrategy = require('passport-twitter').Strategy;

exports.setup = function setupTwitterStrategy(User, config) {
  passport.use(new TwitterStrategy({
    consumerKey: config.clientID,
    consumerSecret: config.clientSecret,
    callbackURL: config.callbackURL,
  },
  function passportCb(token, tokenSecret, profile, done) {
    User.findOne({
      'twitter.id_str': profile.id,
    }, function findCb(err, user) {
      if (err) {
        return done(err);
      }
      if (!user) {
        const newUser = new User(
        {
          name: profile.displayName,
          username: profile.username,
          provider: 'twitter',
          twitter: profile._json,
        });
        
        newUser.save(function saveCb(saveErr) {
          if (saveErr) return done(saveErr);
          return done(saveErr, user);
        });
      }else {
        return done(err, user);
      }
    });
  }) );
};

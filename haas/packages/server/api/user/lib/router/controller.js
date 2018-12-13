 const log = require.main.require('./common/logger');
 /*
  @public
*/
// TODO: improve error handling
function validationError(res, err) {
  return res.status(422).json(err || null);
}

exports = module.exports = function userCtrl(UserModel, auth) {
  const User = UserModel;
  return {publicIndex, index, create, show, destroy, destroyMe, changePassword, me, saveMe };
  /**
   * Get list of users
   * restriction: 'admin'
  */
  function index(req, res) {
    User.find({}, '-salt -hashedPassword', function(err, users) {
      if (err) return res.status(500).send(err);
      res.status(200).json(users);
    });
  }

  /**
   * Get list of users' public data
   * restriction: 'admin'
  */
  function publicIndex(req, res) {
    User.find({}, 'name _id', function(err, users) {
      if (err) return res.status(500).send(err);
      res.status(200).json(users);
    }).lean();
  }

/**
 * Creates a new user
 */
  function create(req, res) {
    const newUser = new User(req.body);
    newUser.provider = 'local';
    newUser.refreshToken = auth.signRefreshToken();

    newUser.save((err, user) => {
      if (err) return validationError(res, err);

      // TODO: find out why serializing entire user won't work...
      auth.sendTokenWithRefresh(auth.signAccessToken(user.token), user.refreshToken, res);
    });
  }

  /**
   * Get a single user
   */
   // TODO: is this safe
  function show(req, res, next) {
    if (!req.params.id) {
      return res.sendStatus(500);
    }

    User.findById(req.params.id, function(err, user) {
      if (err) return next(err);
      if (!user) return res.status(401).end();
      res.json(user.profile);
    }).lean();
  }

  /**
   * Deletes a user
   * restriction: 'admin'
   */
  function destroy(req, res, id) {
    const userID = req.params.id || id;

    if (!userID) {
      return res.sendStatus(500);
    }

    User.findByIdAndRemove(id, function(err, user) {
      if (err) {
        return res.sendStatus(500);
      }
      return res.sendStatus(204);
    });
  }

   /**
   * Deletes a user
   * restriction: 'user'
   */
  function destroyMe(req, res, next) {
    this.destroy(req, res, req.user.id);
  }

  // function sendResetLink(req, res, next) {
  //   const email = req.params.email.toString(36);

  //   log.debug('mail is', email);
  //   if (!email) { return validationError(res, 'No email provided') }

  //   User.findOne({email: email}, function findOneCb(err, user) {
  //     mailer.sendResetEmail(email, parseInt(user.id, 36), user.resetToken);
  //   });

  //   return res.send('Email sent!');
  // }

  // function resetPassword(req, res, next) {
  //   if (!req.body.password) {
  //     return validationError(res, 'No password provided');
  //   }

  //   User.findById(req.user.id, function findByIdCb(err, user) {
  //     if (err) { return validationError(res, err); }
  //     user.password = req.body.password;
  //     user.save(function saveCb(saveErr) {
  //       if (saveErr) return validationError(res, saveErr);
  //       res.status(200).end();
  //     });
  //   })
  // }
  /**
   * Change a users password
   */
  function changePassword(req, res) {
    if(!req.user.id) {
      return res.sendStatus(401);
    }

    const oldPass = String(req.body.oldPassword);
    const newPass = String(req.body.newPassword);

    User.findById(req.user.id, function findByIdCb(err, user) {
      if (err) {
        log.error(err);
        return validationError(res, err);
      }

      if (!user.authenticate(oldPass) ) {
        return res.status(403).end();
      }

      user.password = newPass;
      user.save(function saveCb(saveErr) {
        if (saveErr) {
          log.error(saveErr);
          return validationError(res, saveErr);
        }

        res.status(200).end();
      });
    });
  }

  /**
   * Get my info
   * don't ever give out the password or salt
   * @methods self.userID
   */
  function me(req, res) {
    // if (req.user.role === 'guest') {
    //   // TODO, is returning the res good form? it's redundant no?
    //   return res.json(req.user);
    // }
    if (!req.user.id) {
      // TODO, is returning the res good form? it's redundant no?
      return res.sendStatus(401);
    }

    User.findById(req.user.id, '-salt -hashedPassword -__v', (err, user) => {
      if (err) {
        log.error(err);
        return res.sendStatus(500);
      }

      if (!user) return res.sendStatus(404);

      // TODO, is returning the res good form? it's redundant no?
      return res.json(user.toJSON());
    });
  }

  // Update the user profile
  // TODO: Security....
  function saveMe(req, res) {
    if(!req.user) {
      return _notAuthorized(res);
    }

    console.info('got the request to update user');
    //For now only allow update of name
    // if(!req.body.name) {
    //   return _noJob(res);
    // }

    User.findByIdAndUpdate(req.user.id, req.body, {new: true/*, fields: projection*/})
    .exec((err, updatedUser) => res.json(updatedUser.toJSON()));
    // if (!req.user.id) {
    //   // TODO, is returning the res good form? it's redundant no?
    //   return res.sendStatus(401);
    // }

    // const sentID = req.body._id || req.body.id;

    // if (sentID !== req.user.id) {
    //   // TODO, is returning the res good form? it's redundant no?
    //   return res.sendStatus(403);
    // }

    // User.findByIdAndUpdate(req.user.id, req.body, {new: true}, (err, user) => {
    //   if (err) {
    //     log.error(err);
    //     return res.sendStatus(500);
    //   }

    //   if (!user) return res.sendStatus(404);

    //   // TODO, is returning the res good form? it's redundant no?
    //   return res.json(user.toJSON());
    // });
  }
  /**
   * Authentication callback
   */
  // function authCallback(req, res, next) {
  //   res.redirect('/');
  // },
};

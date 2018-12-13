let UserModel;

exports.setup = function setupGuestStrategy(User) {
  UserModel = User;
};

exports.authenticate = function authenticate(cb) {
  const guest = UserModel.makeGuest();
  let err;

  if (!guest) {
    err = 'no guest token made';
  }

  cb(err, guest);
};


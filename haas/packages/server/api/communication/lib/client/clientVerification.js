const socketioJwt = require('socketio-jwt');
const log = require.main.require('./common/logger');
const jwt = require('jsonwebtoken');

// TODO: Upgrade  tohandle new jwt jwks-rsa scheme
// TODO: all tokens here are just called token, completely because of
// socketio-jwt's hardcoding of the name, fix to be same as decodeToken func
// @param {Object} user
class ClientVerification {
  constructor(socketServer, user, opts) {
    this.secret = user.tokenManager.secret;
    if (!this.secret) {
      throw new Error('client-util must have secret');
    }

    this.server = socketServer;
    this.timeout = (opts && opts.timeout) || 15000; // milliseconds
  }

  authenticate() {
    const self = this;

    this.server
      .on(
        'connection',
        socketioJwt.authorize({
          secret: self.secret,
          timeout: self.timeout // 15 seconds
        })
      )
      .on('authenticated', socket => {
        // socket.decoded_token.id is the user's id
        socket.join(socket.decoded_token.id);

        _manageRoomConnection.call(self, socket);
      });
  }
}

module.exports = ClientVerification;

/* private*/
function _manageRoomConnection(socket) {
  socket.on('loggedIn', data => {
    jwt.decodeToken(data.token, this.secret, (err, decodedToken) => {
      if (err) {
        return log.error('invalid_token', err);
      }

      const userID = decodedToken.id;
      socket.join(userID);
    });
  });

  socket.on('disconnect', () => {
    socket.removeAllListeners('loggedIn');
    socket.removeAllListeners('authenticated');
  });

  socket.on('error', err => {
    log.warn('Socketio error in ClientVerification', err);
  });
}

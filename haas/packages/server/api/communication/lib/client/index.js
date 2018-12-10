const _EventEmitter = require('events').EventEmitter;
const _ = require('lodash');
const _io = require('socket.io');
const _wildcardMiddleware = require('socketio-wildcard');
const _redisSocketIOadapter = require('socket.io-redis');
const _ClientVerification = require('./clientVerification');
const log = require.main.require('./common/logger');

// default socket.io server options
const _default = {
  transports: ['websocket', 'polling', 'xhr-polling', 'jsonp-polling'],
  serveClient: true,
};

class ClientCommunication extends _EventEmitter {
  constructor(httpServer, user, config) {
    if (!user) {
      throw new Error('called ClientComm without user instance module');
    }
    super();

    const opts = config || {};
    opts.middleware = opts.middleware || [];
    opts.middleware.push(_wildcardMiddleware() );

    this.events = _.merge({}, {update: 'update'}, opts.events);
    this.server = _createSocketServer(httpServer, opts);

    const _socketAuth = new _ClientVerification(this.server, user, opts);
    _socketAuth.authenticate();

    this.listen();
  }

  listen() {
    const self = this;
    this.server.on('connection', function onConnection(socket) {
      self.emit('connected', socket);
    });
  }

  send(room, event, message) {
    this.server.to(room).emit(event, JSON.stringify(message) );
  }

  sendAll(event, message) {
    this.server.emit(event, message);
  }

  // message can be object or string
  // socket.io auto encodes to json
  sendSocket(socket, event, message) {
    // send to a single socket connection
    if (!(socket && message) ) {
      // better error handling
      const err = new Error('Error, id or message provided sendOne');
      log.error(err);
      return err;
    }
    socket.emit(event, message );
  }
}

module.exports = ClientCommunication;

/* private*/
function _prepareSocketOpts(opts) {
  return {
    transports: opts.transports || _default.transports,
    serveClient: opts.serveClient || _default.serveClient,
    middleware: opts.middleware || {},
  };
}

function _createSocketServer(httpServer, opts) {
  // TOOD: check that middleware is an array, if not throw error
  const socketServer = _io(httpServer, _prepareSocketOpts(opts) );

  // socketServer.adapter(_redisSocketIOadapter({
  //   pubClient: createConnection
  // }))
  if (opts.middleware) {
    for (let i = 0; i < opts.middleware.length; ++i) {
      socketServer.use(opts.middleware[i] );
    }
  }

  return socketServer;
}

// function _formatMessage(message) {
//   let formattedMessage = message;

//   if (typeof message === 'object') {
//     formattedMessage = JSON.stringify(message);
//   } else if (typeof message !== 'string') {
//     // better error handling
//     // what should we do with this, return nothing to calling func?
//     throw new Error('Message malformed');
//   }
//   return formattedMessage;
// }

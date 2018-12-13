// Communicate with Annotation server
// Based on redis pubsub, but could be ported to others, point is to abstract
const _ = require('lodash');
const _redis = require('redis');
const log = require.main.require('./common/logger');

class ServerComm {
  constructor(config) {
    const commConfig = config.server;

    this.port = commConfig.port;
    this.host = commConfig.host;

    this.connections = {};
    this.createConnection('main');
  }

  // accepts either a nubmer or an array of objects
  createConnection(name) {
    if (this.connections[name]) {
      return this.connections[name];
    }
    this.connections[name] =
     _redis.createClient(this.port, this.host, {detect_buffers: false});

    this.connections[name].on('error', err => log.error(err) );
    return this.connections[name];
  }

  getHash(key, callback) {
    this.connections.main.get(key, function getCb(err, data) {
      callback(err, JSON.parse(data) );
    });
  }

  removeFromList(listName, listItem, callback) {
    const connection = this.createConnection(listName);

    connection.lrem(listName, 0, listItem, callback);
  }

  watchList(queueName, callback) {
    const connection = this.createConnection(queueName);
    const self = this;

    connection.brpop(queueName, 1000, function brpopCb(err, data) {
      process.nextTick((tErr, tData) => {
        callback(tErr, tData);
      }, err, data);
      self.watchList(queueName, callback);
    });
  }
  // expects length 2 queue array
  setListRelay(queueArray, callback) {
    const connection = this.createConnection(queueArray[1] );
    const self = this;

    connection.brpoplpush(queueArray[0], queueArray[1], 0,
    function brpoplpushCb(err, data) {
      process.nextTick((tErr, tData) => {
        return callback(tErr, tData);
      }, err, data);
      self.setListRelay(queueArray, callback);
    });
  }
  
  /*
  *@param {Num|Str} keyExpiration : time in milliseconds for job to live
  */
  setBackedList(nameArray, dataObjArray, keyExpiration, cb) {
    if (!(_.isArray(dataObjArray) && dataObjArray[0] ) ) {
      throw new Error(`dataObjArray must be array w/ >=1 truthy value`);
    } else if (!(_.isArray(nameArray) && nameArray[0] && nameArray[1] ) ) {
      throw new Error('nameArray must have 2 values'); // allows nonsense
    }
    log.debug('expiration is', keyExpiration);

    let keyData;
    if (typeof dataObjArray[0] === 'object') {
      keyData = JSON.stringify(dataObjArray[0] );
    } else {
      keyData = dataObjArray[0];
    }

    let listData;
    if (dataObjArray.length === 1) { // allows falsy in pos 1; up to user
      listData = keyData;
    } else if (typeof dataObjArray[1] === 'object') { // allows null
      listData = JSON.stringify(dataObjArray[1]);
    } else {
      listData = dataObjArray[1];
    }
    // lrem used to avoid strange pileups
    // the key stores a full json representation of the job
    let setData;
    if (keyExpiration > 0) {
      setData = [nameArray[0], keyData, 'PX', keyExpiration]; // expire in ms
    } else {
      setData = [nameArray[0], keyData];
    }

    this.connections.main.multi()
    .set(setData)
    .lrem(nameArray[1], 0, listData)
    .lpush(nameArray[1], listData)
    .exec(function execCb(err, replies) {
      if (err) { log.error(err); }
      return cb(err, replies);
    });
  }

  publish(event, message) {
    this.connections.main.publish(event, message);
  }

  subscribe(event, callback) {
    const connection = this.createConnection(event);
    connection.subscribe(event);

    connection.on('message', (tEvent, message) => {
      process.nextTick((ev, msg) => {
        return callback(ev, msg);
      }, tEvent, message);
    });
  }
  
}

exports = module.exports = ServerComm;

// createConnections(num) {
//     if (_.isNumber(num)) {
//       _(_.range(num) ).forEach(function foreach() {
//         this.freeConnections.push(_redis.createClient(this.port, this.host,
//           {detect_buffers: true}) );
//       });
//       return;
//     }
//     throw new Error('connections property must be a number');
//   }

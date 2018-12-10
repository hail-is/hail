// TODO: add failed
const _EventEmitter = require('events').EventEmitter;
const log = require.main.require('./common/logger');

class JobComm extends _EventEmitter {
  constructor(comm, passedConfig) {
    if (!(comm && passedConfig.comm) ) {
      throw new Error('called JobComm without Comm or config');
    }
    super();

    const config = passedConfig.comm;
    this.client = comm.client;
  }

  /*

  * @param {Obj|Str} data : Obj or JSON
    * @prop {Str} channel : which channel to emit on (optional)
    * @prop {Str} room : the id on the channel to send to
      * (channels can have many rooms, each with unique name)
    * @prop {Str} publicID : the id used to identify the job on the client side
    * @prop {Mixed} data : whatever the sending program wishes to tell client
  */
  notifyUser(event, job, messageData) {
    const mData = this.formatMessage(job, messageData);

    if(!mData) {
      log.error('cannot send message because formatting job failed', job);
      return;
    }
    
    this.client.send(mData.room, event, mData.message);
  }

  formatMessage(jobObj, data) {
    if (jobObj._id === undefined || jobObj.userID === undefined) {
      log.error(new Error("_id and userID required in jobObj"));
      return;
    }

    if (data === undefined) {
      log.error(new Error("no data to send"));
      return;
    }

    return {
      room: jobObj.userID,
      message: {
        _id: jobObj._id,
        data: data,
      },
    };
  }
}

exports = module.exports = JobComm;
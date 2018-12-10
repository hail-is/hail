// TODO: add failed
const yaml = require('js-yaml');
const fs = require('fs');

const configPath = require.main.require('./common/config').jobs.queues.configPath;

// Get document, or throw exception on error
const config = yaml.safeLoad(fs.readFileSync(configPath));

const log = require.main.require('./common/logger');
const Fivebeans = require('fivebeans');

const submitClient = new Fivebeans.client(config.beanstalkd.server, config.beanstalkd.port);

/**************************** Events (watch tubes) ***************************/
const eventsClient = new Fivebeans.client(config.beanstalkd.server, config.beanstalkd.port);

const timeToRun = 60 * 60 * 48; //48 hours in seconds
const priority = 0;
const delay = 1;
const maxReserves = 2;

submitClient.on('connect', () => {
    log.debug('index submitClient client connected');
  })
  .on('error', (err) => {
    throw err;
  })
  .on('close', () => {
    log.info('submitClienttedClient closed');
  })
  .connect();

eventsClient.on('connect', () => {
    log.debug('index eventsClient connected');
  })
  .on('error', (err) => {
    throw err;
  })
  .on('close', () => {
    log.info('eventsClient closed');
  })
  .connect();

submitClient.ignore(['default'], (err) => {
  log.error('failed to ignore default tube in annotationQueue submitClient');
});

submitClient.use(config.beanstalkd.tubes.index.submission, () => {
  log.debug('using submission tube called');
});
submitClient.watch(config.beanstalkd.tubes.index.submission, () => {
  log.debug('watching submission tube');
});

eventsClient.ignore('default', (err) => {
  log.error('failed to ignore default tube in annotationQueue eventsClient');
});
eventsClient.watch(config.beanstalkd.tubes.index.events, () => {
  log.debug('watching started tube');
})
eventsClient.use(config.beanstalkd.tubes.index.events, () => {
  log.debug('using started tube');
})

// const reserveFunc = function() {
//   eventsClient.reserve_with_timeout(10, (err, jobID, payload) => {
//   let job;
//   if(err == 'TIMED_OUT') {
//     return reserveFunc();
//   }
//   console.info('err is ', err);
//   console.info('job id is', jobID);
//   try {
//     job = JSON.parse(payload);
//   } catch (e) {
//     console.info('error', e);
//     eventsClient.destroy(jobID, (err) => {
//       console.info('failed destroying', jobID);
//     });
//   }

//   console.info('reserved in annotationQueue', job);
//   reserveFunc();
// }) };

// reserveFunc();

exports = module.exports = {
  submitClient,
  eventsClient,
  timeToRun,
  priority,
  delay,
  maxReserves,
  events: config.beanstalkd.events,
};

// TODO: decide whether the redis server should persist the jobs itself
// if so, can move back to just listening for jobs.
// subscribeToEvents() {
//   for (const prop in this.events) {
//     if (this.events.hasOwnProperty(prop) ) {
//       this.server.subscribe(this.events[prop], this.notifyUser);
//     }
//   }
// }
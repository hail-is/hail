// TODO: add failed
const yaml = require('js-yaml');
const fs = require('fs');

const configPath = require.main.require('./common/config').jobs.queues
  .configPath;

// Get document, or throw exception on error
const { beanstalkd, tubes } = yaml.safeLoad(fs.readFileSync(configPath));

const log = require.main.require('./common/logger');
const Fivebeans = require('fivebeans');

const submitClient = new Fivebeans.client(beanstalkd.server, beanstalkd.port);

/**************************** Events (watch tubes) ***************************/
const eventsClient = new Fivebeans.client(beanstalkd.server, beanstalkd.port);

const timeToRun = 60 * 60 * 48; //48 hours in seconds
const priority = 0;
const delay = 1;
const maxReserves = 2;

submitClient
  .on('connect', () => {
    log.debug('saveFromQueryQueue submitClient client connected');
  })
  .on('error', err => {
    throw err;
  })
  .on('close', () => {
    log.debug('saveFromQueryQueue submitClienttedClient closed');
  })
  .connect();

eventsClient
  .on('connect', () => {
    log.debug('saveFromQueryQueue eventsClient connected');
  })
  .on('error', err => {
    throw err;
  })
  .on('close', () => {
    log.debug('saveFromQueryQueue eventsClient closed');
  })
  .connect();

submitClient.use(beanstalkd.tubes.saveFromQuery.submission, () => {
  log.debug('using submission tube called');
});
submitClient.watch(beanstalkd.tubes.saveFromQuery.submission, () => {
  log.debug('watching submission tube');
});
submitClient.ignore('default', err => {
  if (err) {
    log.error(err);
  }
});
submitClient.list_tubes_watched((err, tubelist) => {
  if (err) {
    return log.error(err);
  }

  return log.debug(`SaveFromQuery submit client watching: ${tubelist}`);
});

eventsClient.watch(beanstalkd.tubes.saveFromQuery.events, () => {
  log.debug('watching started tube');
});
eventsClient.use(beanstalkd.tubes.saveFromQuery.events, () => {
  log.debug('using started tube');
});
eventsClient.ignore('default', err => {
  if (err) {
    log.error(err);
  }
});
eventsClient.list_tubes_watched((err, tubelist) => {
  if (err) {
    return log.error(err);
  }

  return log.debug(`SaveFromQuery events client watching: ${tubelist}`);
});

exports = module.exports = {
  submitClient,
  eventsClient,
  timeToRun,
  priority,
  delay,
  maxReserves,
  events: beanstalkd.events
};

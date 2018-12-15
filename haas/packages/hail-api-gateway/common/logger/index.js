const winston = require('winston');

winston.handleExceptions([
  new winston.transports.File({
    filename: './logs/exceptions.log',
    colorize: true,
    timestamp: true,
    prettyPrint: true,
    json: true,
    handleExceptions: true,
    humanReadableUnhandledException: true
  }),
  new winston.transports.Console({
    colorize: true,
    timestamp: true,
    prettyPrint: true,
    json: true,
    handleExceptions: true,
    humanReadableUnhandledException: true
  })
]);

// winston.add(winston.transports.File, {
//     filename: 'path/to/all-logs.log',
//     handleExceptions: true,
//   });
const formatter = {
  timestamp: () => Date.now(),
  formatter: options => {
    // - Return string will be passed to logger.
    // - Optionally, use options.colorize(options.level, <string>) to
    //   colorize output based on the log level.
    return (
      options.timestamp() +
      ' ' +
      config.colorize(options.level, options.level.toUpperCase()) +
      ' ' +
      (options.message ? options.message : '') +
      (options.meta && Object.keys(options.meta).length
        ? '\n\t' + JSON.stringify(options.meta)
        : '')
    );
  }
};

const logger = new winston.Logger({
  transports: [
    new winston.transports.File({
      name: 'file.debug',
      level: 'debug', // info, warn, error will log
      filename: './logs/debug.log',
      colorize: true,
      timestamp: true,
      prettyPrint: true,
      json: true
    }),
    new winston.transports.File({
      name: 'file.info',
      level: 'info', // info, warn, error will log
      filename: './logs/info.log',
      colorize: true,
      timestamp: true,
      prettyPrint: true,
      json: true,
      formatter
    }),
    new winston.transports.File({
      name: 'file.error',
      level: 'error', // error will log
      filename: './logs/error.log',
      colorize: true,
      timestamp: true,
      prettyPrint: true,
      json: true
    }),
    new winston.transports.Console({
      level: 'info',
      colorize: true,
      timestamp: true,
      prettyPrint: true,
      json: true
    })
  ],
  colors: {
    trace: 'magenta',
    input: 'grey',
    verbose: 'cyan',
    prompt: 'grey',
    debug: 'blue',
    info: 'green',
    data: 'grey',
    help: 'cyan',
    warn: 'yellow',
    error: 'red'
  }
});

logger.stream = {
  write: message => {
    logger.info(message);
  }
};

// const Loggly = require('winston-loggly').Loggly;
// const loggly_options={ subdomain: "mysubdomain",
// inputToken: "efake000-000d-000e-a000-xfakee000a00" }
// logger.add(Loggly, loggly_options);

// logger.info('Chill Winston, the logs are being
// captured 3 ways- console, file, and Loggly');

module.exports = logger;

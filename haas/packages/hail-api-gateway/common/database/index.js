const Mongoose = require('mongoose');

const dbConfig = require.main.require('./common/config').database; //FIX THIS

const options = dbConfig.options ? `?${dbConfig.options}` : '';

console.info(
  'mongo connection string is',
  `mongodb://${dbConfig.uri}/${dbConfig.database}${options}`
);

Mongoose.connect(
  `mongodb://${dbConfig.uri}/${dbConfig.database}${options}`,
  {
    auth: {
      user: dbConfig.user,
      password: dbConfig.password
    }
  }
);

exports = module.exports = Mongoose;

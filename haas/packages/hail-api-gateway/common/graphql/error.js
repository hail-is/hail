// const { createError } = require('apollo-errors');
const {
  AuthenticationError,
  UserInputError,
  ApolloError
} = require('apollo-server');

class CustomError extends Error {
  constructor(name, message, status, extra) {
    super(message, extra);

    this.name = name;
    this.code = name;
    this.status = status;

    Error.captureStackTrace(this, CustomError);
  }
}

class AuthError {
  constructor(name, message = "The user isn't authenticated", status, extra) {
    return new AuthenticationError(message);
  }
}

class DataFetchError {
  constructor() {
    return new ApolloError("Couldn't fetch the data");
  }
}

module.exports = {
  AuthError,
  DataFetchError
};

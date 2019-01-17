// A singleton that should be configured once
// Manages auth0 calls, fetching access  tokens, and  decoding id tokens
import auth0 from 'auth0-js';
import { string } from 'prop-types';

declare type state = {
  idToken?: string;
  accessToken?: string;
  user?: object;
};

declare type Auth = {
  state: state;
  login: () => void;
  logout: () => void;
};

const Auth = {
  state: {
    idToken: null,
    accessToken: null,
    user: null
  }
};

Auth.

export default Auth;

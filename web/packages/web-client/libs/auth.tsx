// A singleton that should be configured once
// Manages auth0 calls, fetching access  tokens, and  decoding id tokens
import auth0 from 'auth0-js';
import getConfig from 'next/config';

declare type state = {
  idToken: string | null;
  accessToken: string | null;
  user: object | null;
  exp: Date | null;
};

declare type Auth = {
  state: state;
  auth0?: auth0.WebAuth;
  initialize: () => void;
  login: (state: string) => void;
  logout: () => void;
  setState: () => void;
  getState: () => void;
};

const {
  DOMAIN,
  CLIENT_ID,
  RESPONSE_TYPE,
  SCOPE,
  CALLBACK_SUFFIX
} = getConfig().publicRuntimeConfig;

// https://basarat.gitbooks.io/typescript/docs/tips/lazyObjectLiteralInitialization.html
const Auth = {} as Auth;

Auth.state = {
  idToken: null,
  accessToken: null,
  user: null,
  exp: null
};

Auth.initialize = () => {
  if (Auth.auth0) {
    return;
  }

  if (typeof window === 'undefined') {
    console.error('Auth.initialize should be called from client side only');
    return;
  }

  const parts = window.location.href.split('//');

  const redirectUri = `${parts[0]}//${parts[2]}/${CALLBACK_SUFFIX}`;

  console.info('callback', redirectUri);
  Auth.auth0 = new auth0.WebAuth({
    domain: DOMAIN,
    clientID: CLIENT_ID,
    redirectUri: redirectUri,
    responseType: RESPONSE_TYPE || 'token id_token',
    scope: SCOPE || 'openid'
  });
};

Auth.login = state => {
  Auth.auth0.authorize({
    state: this.state,
    prompt: 'login'
  });
};

Auth.setState = () => {};

Auth.getState = () => {};

export default Auth;

// A singleton that should be configured once
// Manages auth0 calls, fetching access  tokens, and  decoding id tokens

// TOOD: Think about safety of using leeway to fudge jwt expiration by N seconds
import auth0 from 'auth0-js';
import getConfig from 'next/config';
import cookies from '../libs/cookies';

declare type user = {
  sub: string;
};

declare type state = {
  idToken: string | null;
  accessToken: string | null;
  user: user | null;
  exp: number | null;
};

declare type Auth = {
  state: state;
  auth0?: auth0.WebAuth;
  hydrate: (state: state) => void;
  handleCallback: () => void;
  initialize: () => void;
  login: (state?: string) => void;
  logout: () => void;
  setState: (result: object) => void;
  getState: () => void;
  clearState: () => void;
  setStateAndCookie: (result: object) => void;
};

const {
  DOMAIN,
  CLIENT_ID,
  RESPONSE_TYPE,
  SCOPE,
  CALLBACK_SUFFIX
} = getConfig().publicRuntimeConfig.AUTH0;

// https://basarat.gitbooks.io/typescript/docs/tips/lazyObjectLiteralInitialization.html
const Auth = {} as Auth;

Auth.state = {
  idToken: null,
  accessToken: null,
  user: null,
  exp: null
};

// Change the reference, so state can be passed by reference and
// shallow watched
Auth.clearState = () => {
  Auth.state = {
    idToken: null,
    accessToken: null,
    user: null,
    exp: null
  };
};

let timeout: NodeJS.Timeout;
const pollForSession = () => {
  if (timeout) {
    clearTimeout(timeout);
  }

  timeout = setTimeout(() => {
    // This boilerplate is required by typescript for union types including null
    // however, there may be a more elegant way to handle this
    if (!Auth.auth0) {
      console.error('Auth library is not initialized in pollFromSession');
      return;
    }

    Auth.auth0.checkSession({}, (err, authResult) => {
      if (err) {
        console.error(err);
        return;
      }

      Auth.setStateAndCookie(authResult);
    });
  }, Math.floor((Auth.state.exp as number) / 2));
};

Auth.initialize = () => {
  if (typeof window === 'undefined') {
    console.error('Auth.initialize should be called from client side only');
    return;
  }

  // split uses a greedy pattern, removing contiguous sequences of /
  const parts = window.location.href.split('/');

  const redirectUri = `${parts[0]}//${parts[2]}/${CALLBACK_SUFFIX}`;

  Auth.auth0 = new auth0.WebAuth({
    domain: DOMAIN,
    clientID: CLIENT_ID,
    redirectUri: redirectUri,
    responseType: RESPONSE_TYPE || 'token id_token',
    scope: SCOPE || 'openid',
    leeway: 2
  });
};

Auth.login = state => {
  const opts: any = { prompt: 'login' };

  if (state) {
    opts.state = state;
  }

  return new Promise((resolve, reject) => {
    if (!Auth.auth0) {
      console.error('Auth library is not initialized');
      return;
    }

    Auth.auth0.popup.authorize(opts, (err, result) => {
      if (err) {
        return reject(err);
      } else {
        Auth.setStateAndCookie(result);
        return resolve(true);
      }
    });
  });
};

Auth.hydrate = state => {
  console.log('state', state);
};

Auth.setStateAndCookie = (result: any) => {
  Auth.setState(result);

  cookies.set('idToken', Auth.state.idToken as string, {
    expires: new Date(Auth.state.exp as number)
  });

  cookies.set('accessToken', Auth.state.accessToken as string, {
    expires: new Date(Auth.state.exp as number)
  });

  pollForSession();
};

Auth.setState = (result: any) => {
  console.info('result', result);

  Auth.state.exp = result.expiresIn * 1000 + new Date().getTime();

  console.info('state timeout', Auth.state.exp);

  Auth.state.accessToken = result.accessToken;
  Auth.state.idToken = result.idToken;

  Auth.state.user = {
    sub: result.idTokenPayload.sub
  };
};

Auth.getState = () => {};

export default Auth;

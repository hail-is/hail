// A singleton that should be configured once
// Manages auth0 calls, fetching access  tokens, and  decoding id tokens

// TOOD: Think about safety of using leeway to fudge jwt expiration by N seconds
import auth0 from 'auth0-js';
import getConfig from 'next/config';
import cookies from 'js-cookie';
import jwtDecode from 'jwt-decode';

declare type user = {
  sub: string;
};

// We store all primary states in cookies
// State derived from the idToken, user, is not stored,
// This may change if we start issuing requests for user profile data
// that is not in idToken
// exp is also stored in a token, to enable a single function setState
// that takes either an auth0 response, or a constructed response from cookeis
// TODO: decide on this: exp is derived from accessToken's claim
declare type state = {
  idToken: string | null;
  accessToken: string | null;
  user: user | null;
  exp: number | null;
  loggedOut: boolean;
};

// TODO: add interface, expose only login, logout, initialize
// TODO: Maybe convert to singleton class
declare type Auth = {
  state: state;
  auth0?: auth0.WebAuth;
  isAuthenticated: () => boolean;
  initialize: () => void;
  login: (state?: string) => void;
  logout: () => void;
  setState: (result: object) => void;
  getState: (cookies?: string) => void;
  clearState: (initState?: Partial<state>) => void;
  setStateAndCookie: (result: object) => void;
  checkSession: (
    cb: (err: auth0.Auth0Error | null, authResult: any) => void
  ) => void;
  getStateSSR: (cookie: string) => void;
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
  exp: null,
  loggedOut: false
};

const setCookies = (state: state) => {
  const opt = {
    expires: new Date(state.exp as number),
    path: '/'
  };

  // TODO: this isn't wholly needed, exp is derived from the accessToken;
  cookies.set('idToken', state.idToken as string, opt);
  cookies.set('accessToken', state.accessToken as string, opt);
  cookies.set('exp', `${state.exp}`, opt);
};

const removeCookies = () => {
  cookies.remove('idToken', { path: '/' });
  cookies.remove('accessToken', { path: '/' });
  cookies.remove('exp', { path: '/' });
};
// Change the reference, so state can be passed by reference and
// shallow watched
Auth.clearState = initState => {
  const base = {
    idToken: null,
    accessToken: null,
    user: null,
    exp: null,
    loggedOut: false
  };

  if (initState) {
    Object.assign(base, initState);
  }

  Auth.state = base;

  removeCookies();
};

Auth.isAuthenticated = () => !!Auth.state.user;

Auth.checkSession = (cb = () => {}) => {
  if (!Auth.auth0) {
    throw new Error('Auth library is not initialized in checkSession');
  }

  Auth.auth0.checkSession({}, (err, authResult) => {
    if (err) {
      cb(err, authResult);

      return;
    }

    Auth.setStateAndCookie(authResult);

    cb(err, authResult);
  });
};

let timeout: NodeJS.Timeout;
const pollForSession = () => {
  if (timeout) {
    clearInterval(timeout);
  }

  timeout = setInterval(() => {
    Auth.checkSession((err, _) => {
      if (err) {
        Auth.logout();
      }
    });
  }, Math.floor((Auth.state.exp as number) / 2));
};

Auth.initialize = () => {
  if (typeof window === 'undefined') {
    throw new Error('Auth.initialize should be called from client side only');
  }

  if (Auth.auth0) {
    console.warn('Auth.initialize should only be called once');
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
      throw new Error('Auth library is not initialized');
    }

    Auth.auth0.popup.authorize(opts, (err, result) => {
      if (err) {
        return reject(err);
      }

      Auth.setStateAndCookie(result);
      return resolve(true);
    });
  });
};

// TODO: Decide on either triggering an event through pre-registered
// callback list, relying on the reference changing to trigger shallow watch in React
// on relying on observables, such as nxjs (which adds 2kb to bundle)
Auth.logout = () => {
  console.info('Logging out');
  Auth.clearState({ loggedOut: true });
};

Auth.setStateAndCookie = (result: any) => {
  Auth.setState(result);

  setCookies(Auth.state);

  pollForSession();
};

Auth.setState = (result: any) => {
  Auth.state.exp = result.expiresIn * 1000 + new Date().getTime();

  Auth.state.accessToken = result.accessToken;
  Auth.state.idToken = result.idToken;

  Auth.state.user = {
    sub: result.idTokenPayload.sub
  };
};

Auth.getStateSSR = cookie => {
  // Strangely cookie === '' doesn't work with non-optional argument
  if (!cookie) {
    return;
  }

  let idTokenIdx = -1;
  let accessTokenIdx = -1;
  let expIdx = -1;

  let idToken = '';
  let accessToken = '';
  let exp = '';
  const parts = cookie.split('; ');

  for (let i = parts.length; i--; ) {
    if (idTokenIdx > -1 && accessTokenIdx > -1 && expIdx > -1) {
      break;
    }

    if (idTokenIdx == -1) {
      idTokenIdx = parts[i].indexOf('idToken=');

      if (idTokenIdx !== -1) {
        idToken = parts[i].substr(idTokenIdx + 8);
        continue;
      }
    }

    if (accessTokenIdx == -1) {
      accessTokenIdx = parts[i].indexOf('accessToken=');

      if (accessTokenIdx !== -1) {
        accessToken = parts[i].substr(accessTokenIdx + 12);
        continue;
      }
    }

    if (expIdx == -1) {
      expIdx = parts[i].indexOf('exp=');

      if (expIdx !== -1) {
        exp = parts[i].substr(expIdx + 4);
        continue;
      }
    }
  }

  if (idToken === '') {
    return;
  }

  try {
    const idTokenPayload = jwtDecode(idToken);

    Auth.setState({
      idToken,
      accessToken,
      exp,
      idTokenPayload
    });
  } catch (e) {
    console.error(e);
    return;
  }
};

// TODO: decide whether we want to validate here, or defer until checkSession
Auth.getState = () => {
  if (!Auth.auth0) {
    throw new Error('Auth library not initialized in getState');
  }

  const result: any = {};
  result.idToken = cookies.get('idToken');

  if (!result.idToken) {
    return;
  }

  result.accessToken = cookies.get('accessToken');
  result.exp = cookies.get('exp');

  try {
    result.idTokenPayload = jwtDecode(result.idToken);

    // Set immediate state, to give any components that need this state
    // the ability to optimistically render that state
    Auth.setState(result);

    // Validate that state, to reduce the likelihood that API calls will fail
    // without user expectation
    setTimeout(() => {
      Auth.checkSession(err => {
        if (err) {
          Auth.logout();
          console.error(err);
          return;
        }

        pollForSession();
      });
    }, 500);
  } catch (e) {
    console.error(e);
  }
};

export default Auth;

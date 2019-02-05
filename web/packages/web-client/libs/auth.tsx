// A singleton that should be configured once
// Manages auth0 calls, fetching access  tokens, and  decoding id tokens

// TODO: Think about safety of using leeway to fudge jwt expiration by N seconds
// TODO: Write tests, simplify race condition handling
// TODO: exp is derived from accessToken's claim, decide if needs keeping
// TODO: File issue with auth0js library regarding rapid refresh, fix
// TODO: Enable SSL / secure on cookies
// TODO: Properly handle issues surround cross-origin cookies
// TODO: Notify the user when we can't verify them, so that log-outs aren't scary
// TODO: Figure out if CSRF protection needed, or if auth0 nonce enough against relay
// TODO: Audit security model
// TODO: Handle safari cross-site tracking issues
// TODO: it's more secure to just patch workshop pass into the user's app_metadata
import auth0 from 'auth0-js';
import getConfig from 'next/config';
import cookies, { CookieAttributes } from 'js-cookie';
import jwtDecode from 'jwt-decode';

// We store all primary states in cookies
// State derived from the idToken, user, is not stored,
// This may change if we start issuing requests for user profile data
// that is not in idToken
// exp is also stored in a token, to enable a single function setState
// that takes either an auth0 response, or a constructed response from cookeis
export interface UserInterface {
  sub: string;
  family_name: string;
  given_name: string;
  name: string;
  nickname: string;
  picture: string;
}

export interface AuthInterface {
  readonly user?: UserInterface;
  readonly idToken?: string;
  readonly accessToken?: string;
  readonly loggedOut: boolean;
}

declare type auth0payload = {
  idToken: string;
  accessToken: string;
  expiresIn: number;
  idTokenPayload: UserInterface;

  // This refers to the state parameter passed to auth0
  // Currently only used for workshop password
  state: string;
};

declare type stateType = {
  user?: UserInterface;
  idToken?: string;
  accessToken?: string;
  expires?: string;
  loggedOut: boolean;
};

enum cookieNames {
  ACCESS_TOKEN = 'access_token',
  ID_TOKEN = 'id_token',
  EXPIRES = 'expires',
  STATE = 'state'
}

declare type cookies = {
  id_token: string;
  // This MUST be named access_token for oauth2 compliance
  // if this token is to be used by any Hail services other than this one
  access_token: string;
  expires: string;
};

declare type cb = (state: AuthInterface) => void;

const cfg = getConfig().publicRuntimeConfig;

const {
  DOMAIN,
  AUDIENCE,
  CLIENT_ID,
  RESPONSE_TYPE,
  SCOPE,
  CALLBACK_SUFFIX
} = cfg.AUTH0;

const HAIL_DOMAIN = cfg.HAIL.DOMAIN;

// https://basarat.gitbooks.io/typescript/docs/tips/lazyObjectLiteralInitialization.html
let _state: stateType = { loggedOut: false };

const Auth: AuthInterface = {
  get user() {
    return _state.user;
  },
  get loggedOut() {
    return _state.loggedOut;
  },
  get idToken() {
    return _state.idToken;
  },
  get accessToken() {
    return _state.accessToken;
  }
};

let _auth0: auth0.WebAuth;

let _sessionPoll: NodeJS.Timeout;

let _checkSessionPromise: Promise<any>;

// splice is slow, and iterating over objects 50x slower than over arrays
// so maintain 2 lists, one with callbacks that may be undefined
// the other to be pop/pushed on only for O(1) overall operations
// and O(2n) memory cost
let listeners: any = [];
let removed: any = [];

const cookieOpts = { path: '/' };
// Different opts because access_token is made accessible to other Hail subdomains
const accessTokenOpts = { path: '/', domain: HAIL_DOMAIN };

// TODO: Should we trigger cb with initial state?
export const addListener = (cb: cb) => {
  if (removed.length) {
    const id = removed.pop();

    listeners[id] = cb;
    return id;
  }

  listeners.push(cb);

  return listeners.length - 1;
};

export const removeListener = (id: number) => {
  listeners[id] = undefined;
  removed.push(id);
};

export default Auth;

export function initialize() {
  if (typeof window === 'undefined') {
    throw new Error('Auth.initialize should be called from client side only');
  }

  if (_auth0) {
    console.warn('Auth.initialize should only be called once');
    return;
  }

  const redirectUri = `${_getBaseUrl()}${CALLBACK_SUFFIX}`;

  _auth0 = new auth0.WebAuth({
    domain: DOMAIN,
    audience: AUDIENCE,
    clientID: CLIENT_ID,
    redirectUri: redirectUri,
    responseType: RESPONSE_TYPE || 'token id_token',
    scope: SCOPE || 'openid',
    leeway: 2
  });

  _initState();
}

// TODO: Re-enable if we wish to switch back to popup mode
// or delete if we're settled on its depracation
// export function login(state?: string) {
//   // If we find logout's don't work properly due to the cookie issue
//   // add back prompt: 'login',
// Stangely popup.authorize type signature does not specify type for options
//   const opts: any = { prompt: 'login', connection: 'google-oauth2' };

//   if (state) {
//     opts.state = state;
//   }

//   return new Promise((resolve, reject) => {
//     if (!_auth0) {
//       throw new Error('Auth library is not initialized');
//     }

//     _auth0.popup.authorize(opts, (err, result) => {
//       if (err) {
//         return reject(err);
//       }

//       _loginFromAuth0(result as auth0payload);
//       return resolve(true);
//     });
//   });
// }

export function login(state?: string) {
  // If we find logout's don't work properly due to the cookie issue
  // add back prompt: 'login',
  const opts: auth0.AuthorizeOptions = {
    prompt: 'login',
    connection: 'google-oauth2'
  };

  if (state) {
    opts.state = state;
  }

  _auth0.authorize(opts);
}

// TODO: investigate failure modes;
// Auth0 documentaiton specifies checks for accessToken and idToken
// but there should be no way to have a 200 without these (since they
// are the only point of authorization)
export function authenticationCallback(
  cb: (err: auth0.Auth0ParseHashError | null, res?: stateType) => void
) {
  _auth0.parseHash((err, authResult) => {
    if (err) {
      return cb(err);
    }

    _loginFromAuth0(authResult as auth0payload);
    cb(null, _state);
  });
}

export function isAuthenticated() {
  return !!_state.user;
}

export async function logout() {
  // Set logout to true immediately, to prevent any future _checkSessions from modifying state
  _clearState({ loggedOut: true });

  try {
    if (_checkSessionPromise) {
      await _checkSessionPromise;
    }
  } finally {
    _auth0.logout({ returnTo: _getBaseUrl() });
  }
}

export function initStateSSR(cookie: string) {
  // Strangely cookie === '' doesn't work with non-optional argument
  if (!cookie) {
    _updateState({ loggedOut: false });
    return;
  }

  let idTokenIdx = -1;
  let accessTokenIdx = -1;
  let expIdx = -1;

  let idToken: string | undefined;
  let accessToken: string | undefined;
  let expires: string | undefined;

  const parts = cookie.split('; ');

  for (let i = parts.length; i--; ) {
    if (idTokenIdx > -1 && accessTokenIdx > -1 && expIdx > -1) {
      break;
    }

    if (idTokenIdx == -1) {
      idTokenIdx = parts[i].indexOf(`${cookieNames.ID_TOKEN}=`);

      if (idTokenIdx !== -1) {
        idToken = parts[i].substr(idTokenIdx + cookieNames.ID_TOKEN.length + 1);
        continue;
      }
    }

    if (accessTokenIdx == -1) {
      accessTokenIdx = parts[i].indexOf(`${cookieNames.ACCESS_TOKEN}=`);

      if (accessTokenIdx !== -1) {
        accessToken = parts[i].substr(
          accessTokenIdx + cookieNames.ACCESS_TOKEN.length + 1
        );
        continue;
      }
    }

    if (expIdx == -1) {
      expIdx = parts[i].indexOf(`${cookieNames.EXPIRES}=`);

      if (expIdx !== -1) {
        expires = parts[i].substr(expIdx + cookieNames.ID_TOKEN.length + 1);
        continue;
      }
    }
  }

  if (!idToken) {
    return;
  }

  let user: UserInterface | undefined;
  try {
    user = jwtDecode(idToken) as UserInterface;
  } catch (e) {
    console.error(e);
  }

  // Undefined depending on what fails
  _updateState({
    user,
    idToken,
    accessToken,
    expires,
    loggedOut: false
  });
}

function _initState() {
  if (!_auth0) {
    throw new Error('Auth library not initialized in getState');
  }

  const idToken = cookies.get(cookieNames.ID_TOKEN);

  if (!idToken) {
    _updateState({ loggedOut: false });
    return;
  }

  try {
    // Set immediate state, to give any components that need this state
    // the ability to optimistically render that state
    _updateState({
      idToken: idToken,
      user: jwtDecode(idToken) as UserInterface,
      accessToken: cookies.get(cookieNames.ACCESS_TOKEN),
      expires: cookies.get(cookieNames.EXPIRES),

      loggedOut: false
    });

    _pollForSession();

    // Re-enable if we wish to check the user's session shortly after page refresh
    // This has, currently, the unfortunate side-effect of auto-logging the user out

    // Validate that state, to reduce the likelihood that API calls will fail
    // without user expectation
    // setTimeout(() => {
    //   if (_state.loggedOut) {
    //     return;
    //   }

    //   _checkSession()
    //     .then(() => {
    //       if (!_state.loggedOut) {
    //         _pollForSession();
    //       }
    //     })
    //     .catch((err: any) => {
    //       console.error(err);

    //       // If we logout here safari breaks unless users remove cross-site trackign
    //       logout();
    //     });
    // }, 500);
  } catch (e) {
    console.error(e);
  }
}

function _loginFromAuth0(r: auth0payload) {
  const expMs = `${r.expiresIn * 1000 + Date.now()}`;
  // https://github.com/js-cookie/js-cookie/wiki/Frequently-Asked-Questions#expire-cookies-in-less-than-a-day
  const expires = new Date(expMs);

  // js-cookie accepts a number in ms
  const tokenOpts: CookieAttributes = Object.assign({ expires }, cookieOpts);

  const accessTokenOpt: CookieAttributes = Object.assign(
    { expires },
    accessTokenOpts
  );

  cookies.set(cookieNames.ACCESS_TOKEN, r.accessToken, accessTokenOpt);
  cookies.set(cookieNames.ID_TOKEN, r.idToken, tokenOpts);
  cookies.set(cookieNames.EXPIRES, expMs, tokenOpts);
  cookies.set(cookieNames.STATE, r.state, tokenOpts);

  _updateState({
    accessToken: r.accessToken,
    idToken: r.idToken,
    user: Object.assign({}, r.idTokenPayload),
    expires: expMs,
    loggedOut: false
  });
}

function _removeCookies() {
  cookies.remove(cookieNames.ACCESS_TOKEN, accessTokenOpts);
  cookies.remove(cookieNames.ID_TOKEN, cookieOpts);
  cookies.remove(cookieNames.EXPIRES, cookieOpts);
  cookies.remove(cookieNames.STATE, cookieOpts);
}

// Change the reference, so state can be passed by reference and
// shallow watched
function _clearState(initState?: Partial<stateType>) {
  if (_sessionPoll) {
    clearInterval(_sessionPoll);
  }

  const base = {
    loggedOut: false
  };

  if (initState) {
    Object.assign(base, initState);
  }

  _removeCookies();

  _updateState(base);
}

// This can cause a subtle issue: checkSession sets a cookie briefly
// this cookie may not be cleared if user refreshes browser before cookie cleared
// causing accumulation of garbage
function _checkSession() {
  if (!_auth0) {
    throw new Error('Auth library is not initialized in checkSession');
  }

  const state = cookies.get(cookieNames.STATE);

  _checkSessionPromise = new Promise((resolve, reject) => {
    if (_state.loggedOut) {
      return resolve();
    }

    _auth0.checkSession({ state }, (err, authResult: auth0payload) => {
      if (err) {
        return reject(err);
      }

      if (_state.loggedOut) {
        return resolve();
      }

      _loginFromAuth0(authResult);

      resolve(authResult);
    });
  });

  return _checkSessionPromise;
}

function _pollForSession() {
  if (_sessionPoll) {
    clearInterval(_sessionPoll);
  }

  if (!_state.expires) {
    console.error('_pollForSession must have truthy expiration value');
    return;
  }

  let expIn = Math.floor(
    (parseInt(_state.expires as string, 10) - Date.now()) / 2
  );

  if (isNaN(expIn)) {
    console.error("Couldn't convert expires to number", _state.expires);
    return;
  }

  if (expIn < 0) {
    expIn = 0;
  }

  _sessionPoll = setInterval(() => {
    if (_state.loggedOut) {
      clearInterval(_sessionPoll);
      return;
    }

    _checkSession()
      .then(() => {
        if (_state.loggedOut) {
          clearInterval(_sessionPoll);
        }
      })
      .catch((err: any) => {
        console.error(err);
        logout();
      });
  }, expIn);
}

// expects to receive a new referece;
function _updateState(state: stateType) {
  // const oldState = Object.assign({}, _state);
  _state = state;

  _triggerUpdates();
}

function _triggerUpdates() {
  listeners.forEach((cb?: cb) => {
    if (cb) {
      cb(Auth);
    }
  });
}

var baseUrl: string;
function _getBaseUrl() {
  if (baseUrl) {
    return baseUrl;
  }

  if (typeof window === 'undefined') {
    console.error('_getBaseUrl should only be called in client context');
    return;
  }
  const parts = window.location.href.split('/');

  baseUrl = `${parts[0]}//${parts[2]}/`;

  return baseUrl;
}

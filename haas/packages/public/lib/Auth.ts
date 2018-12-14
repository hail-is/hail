// TODO: Naming conventions; settle on more consistent capitalization rule
// TODO: Make promise-based version of all async
import getConfig from 'next/config';
import { getCookie, setCookie, removeCookie } from '../lib/cookie';

import auth0 from 'auth0-js';
import jwtDecode from 'jwt-decode';

import { store } from 'react-easy-state';

declare type authResult = {
  expiresIn: number;
  accessToken: string;
  idToken: string;
};

declare type authCallback = (
  error: auth0.Auth0ParseHashError | null,
  state: state
) => void;

declare type authProfileCallback = (
  error: auth0.Auth0Error | Error | null,
  profile: object | null
) => void;

declare type user = {
  name: string | null;
  email: string | null;
  picture: string | null;
  sub: string;
  iss: string;
  aud: string;
  exp: number;
  ia: number;
};

declare type state = {
  exp: number | null;
  idToken: string | null;
  accessToken: string | null;
  user: user | null;
  userProfile: object | null;
  loggedOutReason: string | null;
};

declare type Auth = {
  state: state;
  handleAuthenticationAsync(cb: authCallback): void;
  getSetUserProfileAsync(cb: authProfileCallback): void;
  getAccessToken(req?: any): string | null;
  getIdToken(req?: any): string | null;
  getUserID(): string | null;
  login(): void;
  logout(reason?: string): void;
  initialize(req?: any | undefined): void;
  isAuthenticated(): boolean;
  wasAuthenticated(): boolean;
  logoutIfExpired(): void;
  hydrate(state: state): void;
};

const keys = {
  accessToken: 'access_token',
  idToken: 'id_token',
  user: 'user',
  exp: 'expires_at'
};

// getConfig reads from next.config.js
const { publicRuntimeConfig } = getConfig();

const auth0instance = new auth0.WebAuth({
  domain: publicRuntimeConfig.AUTH0.DOMAIN,
  clientID: publicRuntimeConfig.AUTH0.CLIENT_ID,
  redirectUri: publicRuntimeConfig.AUTH0.REDIRECT_URI,
  responseType: publicRuntimeConfig.AUTH0.RESPONSE_TYPE,
  audience: publicRuntimeConfig.AUTH0.AUDIENCE,
  scope: publicRuntimeConfig.AUTH0.SCOPE
});

console.info('audience ', publicRuntimeConfig.AUTH0.AUDIENCE);
const Auth = {} as Auth;

Auth.state = store({
  exp: null,
  idToken: null,
  accessToken: null,
  user: null,
  userProfile: null,
  loggedOutReason: null
});

let renewTimeout: NodeJS.Timeout;

const clearState = () => {
  removeCookie(keys.accessToken);
  removeCookie(keys.idToken);
  removeCookie(keys.exp);

  Auth.state.exp = null;
  Auth.state.idToken = null;
  Auth.state.accessToken = null;
  Auth.state.user = null;
  Auth.state.userProfile = null;

  if (renewTimeout) {
    clearTimeout(renewTimeout);
  }
};

// Hydrate from server state
Auth.hydrate = (state: state) => {
  Auth.state.accessToken = state.accessToken;
  Auth.state.idToken = state.idToken;
  Auth.state.exp = state.exp;
  Auth.state.user =
    state.user || state.idToken ? jwtDecode(state.idToken) : null;
};

// TODO: Memoize without confusing purpose
Auth.getAccessToken = req => {
  if (Auth.state.accessToken) {
    return Auth.state.accessToken;
  }

  return getCookie(keys.accessToken, req);
};

Auth.getUserID = () => {
  if (!Auth.state.user) {
    return null;
  }

  return Auth.state.user.sub;
};

Auth.getSetUserProfileAsync = cb => {
  if (Auth.state.userProfile) {
    cb(null, Auth.state.userProfile);
    return;
  }

  if (!Auth.state.accessToken) {
    cb(new Error('Must be logged in'), null);
    return;
  }

  const auth0Manage = new auth0.Management({
    domain: publicRuntimeConfig.AUTH0.DOMAIN,
    token: Auth.state.accessToken
  });

  auth0Manage.getUser(Auth.state.user.sub, (err, profile) => {
    if (err) {
      console.error(err);
      cb(err, null);
      return;
    }

    console.info('GOT', profile);

    if (profile) {
      Auth.state.userProfile = profile;
    }

    cb(err, Auth.state.userProfile);
  });
};

Auth.getIdToken = req => {
  if (Auth.state.idToken) {
    return Auth.state.idToken;
  }

  return getCookie(keys.idToken, req);
};

Auth.login = () => {
  auth0instance.authorize();
};

Auth.handleAuthenticationAsync = (cb: authCallback) => {
  auth0instance.parseHash((err, authResult: any) => {
    if (authResult && authResult.accessToken && authResult.idToken) {
      setSession(authResult);
    } else if (err) {
      console.log(err);
    }

    cb(err, Auth.state);
  });
};

Auth.initialize = req => {
  setStateOrLogout(req);
};

Auth.logoutIfExpired = () => {
  if (Auth.wasAuthenticated()) {
    Auth.logout();
  }
};

Auth.wasAuthenticated = () => {
  if (Auth.state.user && Auth.state.exp < new Date().getTime()) {
    return true;
  }

  return false;
};

Auth.isAuthenticated = () => {
  if (Auth.state.user && Auth.state.exp >= new Date().getTime()) {
    return true;
  }

  return false;
};

Auth.logout = (reason?: string): void => {
  clearState();

  Auth.state.loggedOutReason = reason || 'Session expired';
};

// declare function setState(x: any): null;
function setStateOrLogout(req?: any): Error | void {
  try {
    const idToken = getCookie(keys.idToken, req);

    Auth.state = {
      exp: getCookie(keys.exp, req),
      accessToken: getCookie(keys.accessToken, req),
      idToken: idToken,
      user: idToken ? jwtDecode(idToken) : null,
      loggedOutReason: null,
      userProfile: null
    };
    console.info('auth state', Auth.state);
    setRenewal(Auth.state.exp);
  } catch (err) {
    console.error('error setting state', err);
    Auth.logout(err.message);

    return err;
  }

  return;
}

function setSession(authResult: authResult) {
  if (!authResult) {
    console.error('No authResult provided');
    return;
  }

  clearState();

  const expiresAt = authResult.expiresIn * 1000 + new Date().getTime();

  setCookie(keys.accessToken, authResult.accessToken, expiresAt);
  setCookie(keys.idToken, authResult.idToken, expiresAt);
  setCookie(keys.exp, String(expiresAt), expiresAt);

  setStateOrLogout();
}

function setRenewal(exp) {
  if (!process.browser) {
    return;
  }

  if (!exp) {
    return;
  }

  if (renewTimeout) {
    clearTimeout(renewTimeout);
  }

  const cTime = new Date().getTime();
  const renewTime = exp > cTime ? Math.floor((exp - cTime) / 2) : 0;

  renewTimeout = setTimeout(() => {
    auth0instance.checkSession({}, (err, result) => {
      if (err) {
        console.error(err);
        return;
      }

      if (result) {
        setSession(result);
      }
    });
  }, renewTime);
}

export default Auth;

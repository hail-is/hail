// TODO: Naming conventions; settle on more consistent capitalization rule
// TODO: Make promise-based version of all async
import getConfig from 'next/config';
import { getCookie, setCookie, removeCookie } from '../lib/cookie';

import auth0 from 'auth0-js';
import jwtDecode from 'jwt-decode';

// Re-enable if we want to watch Auth state
// import { store } from 'react-easy-state';

// getConfig reads from next.config.js
const { publicRuntimeConfig } = getConfig();

const {
  DOMAIN,
  CLIENT_ID,
  RESPONSE_TYPE,
  AUDIENCE,
  SCOPE,
  REDIRECT_URI // we will set this dynamically at initialization if not provided
} = publicRuntimeConfig.AUTH0;

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
  auth0instance: auth0.WebAuth;
  handleAuthenticationAsync(cb: authCallback): void;
  getSetUserProfileAsync(cb: authProfileCallback): void;
  getAccessToken(req?: any): string | null;
  getIdToken(req?: any): string | null;
  getUserID(): string | null;
  login(state?: string): void;
  logout(reason?: string): void;
  initializeState(req?: any): void;
  initializeClient(): void;
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

const Auth = {} as Auth;

Auth.initializeClient = () => {
  let computedUri;
  if (!REDIRECT_URI) {
    const parts = window.location.href.split('/');
    computedUri = `${parts[0]}//${parts[2]}/auth0callback`;
  } else {
    computedUri = REDIRECT_URI;
  }

  Auth.auth0instance = new auth0.WebAuth({
    domain: DOMAIN,
    clientID: CLIENT_ID,
    redirectUri: computedUri,
    responseType: RESPONSE_TYPE,
    audience: AUDIENCE,
    scope: SCOPE
  });
};

Auth.initializeState = req => {
  setStateOrLogout(req);
};

Auth.state = {
  exp: null,
  idToken: null,
  accessToken: null,
  user: null,
  userProfile: null,
  loggedOutReason: null
};

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

Auth.login = state => {
  if (state) {
    Auth.auth0instance.authorize({
      prompt: 'login',
      state
    });
  } else {
    Auth.auth0instance.authorize({
      prompt: 'login'
    });
  }
};

Auth.handleAuthenticationAsync = (cb: authCallback) => {
  Auth.auth0instance.parseHash((err, authResult: any) => {
    if (authResult && authResult.accessToken && authResult.idToken) {
      setSession(authResult);
    }

    cb(err, Auth.state);
  });
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
    Auth.auth0instance.checkSession({}, (err, result) => {
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

import jsCookie from 'js-cookie';

// TODO: set secure: true
// TODO: Finish cookie-based auth
// TODO: is it a security issue to set the cookie client side?
// TODO: we can't use httponly cookies while also implementing
// TODO: make request object pass with a type that has on headers, the cookie
// checkSession....find an alternative to checkSession
export const setCookie = (
  key: string,
  value: string | object,
  expiration?: number
) => {
  console.info('setting cookie', key, value, expiration);
  jsCookie.set(key, value, {
    expires: expiration,
    path: '/'
  });
};

export const removeCookie = (key: string) => {
  console.info('remove cookie', key);
  jsCookie.remove(key, {
    path: '/'
  });
};

export const getCookieFromBrowser = (key: string) => {
  // console.log('grabbing key from browser', jsCookie.get(key));
  return jsCookie.get(key);
};

export const getCookieFromServer = (key: string, req: any) => {
  // console.info('before!', req.headers);ıbvb
  if (!(req.headers && req.headers.cookie)) {
    return undefined;
  }

  const rawCookie = req.headers.cookie
    .split(';')
    .find(c => c.trim().startsWith(`${key}=`));
  if (!rawCookie) {
    return undefined;
  }

  return rawCookie.split('=')[1];

  // return cookie.parse(req.headers.cookie)[key];     √ √Ç
};

export const getCookie = (key: string, req?: any) => {
  return req === undefined
    ? getCookieFromBrowser(key)
    : getCookieFromServer(key, req);
};

// export {
//   setCookie,
//   removeCookie,
//   getCookieFromBrowser,
//   getCookieFromServer,
//   getCookie
// };

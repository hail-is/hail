import jsCookie from 'js-cookie';

const secure = !process.env.NOT_SECURE_COOKIE;
const path = '/';

const defaultOpts = {
  path,
  secure
};

const Cookies = {
  get: (name: string) => jsCookie.get(name),
  set: (name: string, val: string) => jsCookie.set(name, val, defaultOpts),
  remove: (name: string) => jsCookie.remove(name, { path })
};

export default Cookies;

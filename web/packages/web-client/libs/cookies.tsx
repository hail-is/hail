//  This package abstracts away server/client cookie setting/getting differences
import jsCookie from 'js-cookie';

const secure = !process.env.NOT_SECURE_COOKIE;
const path = '/';

const defaultOpts = {
  path,
  secure
};

const Cookies = {
  get: (name: string) => jsCookie.get(name),
  set: (name: string, val: string, opts: {}) =>
    jsCookie.set(name, val, Object.assign({}, defaultOpts, opts)),
  remove: (name: string) => jsCookie.remove(name, { path })
};

export default Cookies;

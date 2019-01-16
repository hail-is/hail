import Auth from 'lib/Auth';

const loginFn = () => Auth.login();

const Link = () => (
  <a color="default" label="Log In" className="link-button" onClick={loginFn}>
    Login
  </a>
);

export default Link;

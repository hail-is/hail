import Auth from 'lib/Auth';

const loginAction = () => Auth.login();

const Link = () => (
  <a
    color="default"
    label="Log In"
    className="link-button"
    onClick={loginAction}
  >
    Login
  </a>
);

export default Link;

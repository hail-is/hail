import App, { Container } from 'next/app';
import Header from '../components/Header';
import { initialize, isAuthenticated, initStateSSR } from '../libs/auth';
import Router from 'next/router';
import jscookies from 'js-cookie';
import 'styles/main.scss';
import 'animate.css';
import {
  Notebook,
  startRequest,
  startListener
} from '../components/Notebook/datastore';

// TODO: think about order of initialization of auth module
// TODO: set some kind of protected property on routes, instead of
// blacklisting here
const protectedRoute: any = {
  '/notebook': true
};

const isServer = typeof window === 'undefined';

let NodeCookies: any;
if (isServer) {
  NodeCookies = require('cookies');
}

declare type props = {
  Component: any;
  ctx: any;
};
// let authInitialized = false;
// TODO: think about using React context to pass down auth state instead of prop
export default class MyApp extends App<props> {
  state = {
    isDark: false
  };

  // Constructor runs before getInitialProps
  constructor(props: any) {
    super(props);

    if (!isServer) {
      // This must happen first, so that child components that use auth
      // in constructor may do so
      initialize();

      if (isAuthenticated() && !Notebook.initialized) {
        startRequest().then(() => startListener());
      }
    }

    this.state.isDark = props.isDark;
  }

  // This runs:
  // 1) SSR Mode: One time
  // 2) Client mode: After constructor (on client-side route transitions)

  static async getInitialProps({ Component, ctx }: props) {
    let pageProps: any = {};
    let isDark = false;

    if (isServer) {
      // Run here because we have no access to ctx in constructor
      initStateSSR(ctx.req.headers.cookie);

      isDark =
        ctx.req.headers.cookie &&
        ctx.req.headers.cookie.indexOf('is_dark=1') > -1;
    } else {
      isDark = !!jscookies.get('is_dark');
    }

    if (!isAuthenticated() && protectedRoute[ctx.pathname] === true) {
      // ctx only exists only on server
      if (ctx.res) {
        const cookies = new NodeCookies(ctx.req, ctx.res);
        cookies.set('referrer', ctx.pathname);

        ctx.res.writeHead(303, { Location: '/login?redirect=true' });
        ctx.res.end();

        return { pageProps, isDark };
      }

      jscookies.set('referrer', ctx.pathname);
      Router.replace(`/login?redirect=true`);

      return { pageProps, isDark };
    }

    if (Component.getInitialProps) {
      pageProps = await Component.getInitialProps(ctx);
    }

    return { pageProps, isDark };
  }

  onDarkToggle = () => {
    this.setState((prevState: any) => {
      if (!prevState.isDark) {
        jscookies.set('is_dark', '1', { path: '/' });

        return { isDark: true };
      }

      jscookies.remove('is_dark', { path: '/' });
      return { isDark: false };
    });
  };

  render() {
    const { Component, pageProps } = this.props;

    return (
      <Container>
        <span id="theme-site" className={this.state.isDark ? 'dark' : ''}>
          <Header />
          <span id="main">
            <Component {...pageProps} />
          </span>

          <span id="footer">
            <i
              className="material-icons"
              style={{ cursor: 'pointer', fontSize: '1rem' }}
              onClick={this.onDarkToggle}
            >
              wb_sunny
            </i>
          </span>
        </span>
      </Container>
    );
  }
}

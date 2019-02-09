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

// let authInitialized = false;
// TODO: think about using React context to pass down auth state instead of prop
export default class MyApp extends App {
  state = {
    isDark: false
  };

  // This runs:
  // 1) SSR Mode: One time
  // 2) Client mode: After constructor (on client-side route transitions)

  static async getInitialProps({
    Component,
    ctx
  }: {
    Component: any;
    ctx: any;
  }) {
    let pageProps: any = {};
    let isDark = false;

    if (isServer) {
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

  constructor(props: any) {
    super(props);

    if (!isServer) {
      initialize();

      if (isAuthenticated() && !Notebook.initialized) {
        startRequest().then(() => startListener());
      }
    }

    this.state.isDark = props.isDark;
  }

  onComponentDidMount() {}
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

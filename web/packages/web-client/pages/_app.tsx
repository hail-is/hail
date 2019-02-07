import App, { Container } from 'next/app';
import Header from '../components/Header';
import { initialize, isAuthenticated, initStateSSR } from '../libs/auth';
import Router from 'next/router';
import cookies from 'js-cookie';

// import cookies from '../libs/cookies';

import 'styles/main.scss';
import 'animate.css';
// import 'normalize.css';

// TODO: set some kind of protected property on routes, instead of
// blacklisting here
const protectedRoute: any = {
  '/notebook': true
};

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

    if (typeof window === 'undefined') {
      initStateSSR(ctx.req.headers.cookie);

      isDark =
        ctx.req.headers.cookie &&
        ctx.req.headers.cookie.indexOf('is_dark') > -1;
    } else {
      isDark = !!cookies.get('is_dark');
    }

    // ctx.pathname will not include get variables in the query
    // will include the full directory path /path/to/resource
    if (!isAuthenticated() && protectedRoute[ctx.pathname] === true) {
      if (ctx.res) {
        ctx.res.writeHead(303, { Location: '/login?redirect=true' });
        ctx.res.end();
        return { pageProps, isDark };
      }

      Router.replace('/login?redirect=true');
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
        cookies.set('is_dark', '1', { path: '/' });

        return { isDark: true };
      }

      cookies.remove('is_dark', { path: '/' });
      return { isDark: false };
    });
  };

  constructor(props: any) {
    super(props);

    this.state.isDark = props.isDark;

    if (typeof window !== 'undefined') {
      initialize();
    }
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
              style={{ cursor: 'pointer' }}
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

import App, { Container } from 'next/app';
import Header from '../components/Header';
import auth from '../libs/auth';
// import cookies from '../libs/cookies';

import 'styles/main.scss';
import 'animate.css';
// TODO: think about using React context to pass down auth state instead of prop
export default class MyApp extends App {
  state = {
    isDark: false
  };

  static async getInitialProps({
    Component,
    ctx
  }: {
    Component: any;
    ctx: any;
  }) {
    let pageProps = {};

    if (typeof window === 'undefined') {
      auth.getStateSSR(ctx.req.headers.cookie);
    }

    if (Component.getInitialProps) {
      pageProps = await Component.getInitialProps(ctx);
    }

    return { pageProps };
  }

  onDarkToggle = () => {
    this.setState((prevState: any) => {
      return {
        isDark: !prevState.isDark
      };
    });
  };

  constructor(props: any) {
    super(props);

    // TOOD: For any components that need to fetch during server phase
    // we will need to extract the accessToken
    if (typeof window !== 'undefined') {
      auth.initialize();
      auth.getState();
    }
  }

  render() {
    const { Component, pageProps } = this.props;

    return (
      <Container>
        <span id="theme-site" className={this.state.isDark ? 'dark' : ''}>
          <Header authState={auth.state} />
          <span id="main">
            <Component {...pageProps} authState={auth.state} />
          </span>

          <span id="footer" style={{ padding: 7, display: 'flex' }}>
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

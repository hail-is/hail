import React from 'react';
import App, { Container } from 'next/app';
import Header from '../components/Header';
import auth from '../libs/auth';
// import cookies from '../libs/cookies';

import 'styles/main.scss';
import 'animate.css';
// TODO: properly handle nextjs, react props types here
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

    if (Component.getInitialProps) {
      pageProps = await Component.getInitialProps(ctx);
    }

    if (typeof window === 'undefined') {
      console.info('stuff', ctx.req.headers.cookie);
    }

    return { pageProps };
  }

  constructor(props: any) {
    super(props);

    if (typeof window !== 'undefined') {
      // Auth initialization logic
      auth.initialize();

      if (props.authState) {
        auth.hydrate(props.hydrateState);
      }
    }
  }

  render() {
    const { Component, pageProps } = this.props;

    return (
      <Container>
        <span id="theme-site" className={this.state.isDark ? 'dark' : ''}>
          <Header />
          <span id="main">
            <Component {...pageProps} />
          </span>
        </span>
      </Container>
    );
  }
}

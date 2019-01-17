import React from 'react';
import App, { Container } from 'next/app';
import { isClientSide } from '../libs/utils';
import Header from '../components/Header';
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

    return { pageProps };
  }

  constructor(props: any) {
    super(props);

    if (isClientSide()) {
      // Auth initialization logic
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

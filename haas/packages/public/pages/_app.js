// TODO: remove unneeded dependencies
// TODO: improve documentation, comments
import Auth from 'lib/Auth';
import Header from 'components/Header';

import App, { Container } from 'next/app';

import { getCookieFromServer, removeCookie, setCookie } from 'lib/cookie';

import Router from 'next/router';

// At least 1 css file, even empty, must be included in _app.js
// unfortunately, or css-loader will cause issues with Link
import 'styles/main.scss';

class Hail extends App {
  // Data returned from getInitialProps is serialized when server rendering,
  // similar to a JSON.stringify.
  // Make sure the returned object from getInitialProps is a plain Object and not using Date, Map or Set.
  // Note: getInitialProps can not be used in children components. Only in pages.
  static async getInitialProps({ Component, ctx }) {
    // Auth.state gets initialized only  once, only in _app
    let pageProps;
    if (Component.getInitialProps) {
      pageProps = await Component.getInitialProps(ctx);
    }

    // Initialize isn't idempotent; it
    if (ctx.req) {
      Auth.initialize(ctx.req);
    }

    // We could also pass auth state through context
    // problem is that we will lose reference to this when
    // transitioning from server to client (namely the es6 store)
    // ctx.authState = Auth.state;

    return {
      pageProps,
      authState: Auth.state,
      darkTheme: !!ctx.req && getCookieFromServer('darkTheme', ctx.req)
    };
  }

  constructor(props, context) {
    super(...arguments);

    if (props.authState) {
      Auth.hydrate(props.authState);
    } else {
      Auth.initialize();
    }

    this.state = { darkTheme: props.darkTheme };
  }

  // ES7-style arrow class method; to provide object-referencing "this"
  handleLogin = () => {
    Auth.login();
    // this.props.apolloClient.resetStore();
    // Or can do
    // const apollo = initApollo();
    // apollo.resetStore();

    Router.push('/');
  };

  handleLogout = () => {
    Auth.logout();
    // this.props.apolloClient.clearStore();
    // Or can do:
    // const apollo = initApollo();
    // apollo.clearStore();

    // May want to check if router on / page
    Router.push('/');
  };

  toggleTheme = () => {
    const dark = !this.state.darkTheme;

    this.setState({ darkTheme: dark });

    if (!dark) {
      removeCookie('darkTheme');
    } else {
      setCookie('darkTheme', true);
    }
  };

  componentDidMount = () => {
    Router.events.on('routeChangeComplete', this.handleRouteChange);
  };

  componentWillUnmount = () => {
    Router.events.off('routeChangeComplete', this.handleRouteChange);
  };

  handleRouteChange = url => {
    // TODO: Notified that you're logged out
    if (Auth.wasAuthenticated()) {
      this.setState({
        openLogoutSnackbar: true
      });

      this.handleLogout();
    }
  };

  render() {
    const { Component, pageProps, apolloClient } = this.props;
    return (
      <Container>
        {/* <Snackbar
          anchorOrigin={{
            vertical: 'top',
            horizontal: 'right'
          }}
          open={this.state.openLogoutSnackbar}
          autoHideDuration={6000}
          onClose={this.handleClose}
        >
          <MySnackbarContentWrapper
            onClose={() => this.setState({ openLogoutSnackbar: false })}
            variant="info"
            message="Logged out (your session expired)"
          />
        </Snackbar> */}

        <div
          className={`theme-site ${this.state.darkTheme ? 'theme-dark' : ''}`}
        >
          <Header
            onLogin={this.handleLogin}
            onLogout={this.handleLogout}
            auth={Auth}
          />
          <div id="_app">
            <Component
              pageContext={this.pageContext}
              {...pageProps}
              auth={Auth}
            />
          </div>
          <i className="material-icons icon-button" onClick={this.toggleTheme}>
            brightness_medium
          </i>
        </div>
      </Container>
    );
  }
}

export default Hail;

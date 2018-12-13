// Injects apollo client into applicaiton context
// React context: https://reactjs.org/docs/context.html
import { Component } from 'react';
import initApollo from './initApollo';
import Head from 'next/head';
import { getDataFromTree } from 'react-apollo';
import PropTypes from 'prop-types';
import Auth from './Auth';

const apolloConfig = { getToken: Auth.getAccessToken };

export default App => {
  return class WithApollo extends Component {
    static displayName = `withApollo(${App.displayName})`;

    static propTypes = {
      apolloState: PropTypes.object.isRequired
    };

    static async getInitialProps(ctx) {
      const {
        Component,
        router,
        ctx: { req, res }
      } = ctx;

      // For some reason, initialization screws us up...
      // We are somehow modifying expectations in Header
      // If we are logged in server-side...
      // May have to do with state management
      // And maybe should build a HOC for authentication

      // Auth.initialize(req);
      const accessToken = Auth.getAccessToken(req);

      const apollo = initApollo(
        {},
        Object.assign({}, apolloConfig, {
          initialAccessToken: accessToken
        })
      );

      // first arg is state, I guess at this point we have no state,
      // but i think maybe could grab from local storage in the future,
      // if we decide to cache for offline use
      ctx.ctx.apolloClient = apollo;

      let appProps = {};
      if (App.getInitialProps) {
        appProps = await App.getInitialProps(ctx);
      }

      if (res && res.finished) {
        // When redirecting, the response is finished, no point in continuing
        // to render
        return {};
      }
      // Run all GraphQL queries in the component tree
      // and extract the resulting data
      // initApollo is initialized in the constructor
      // after which it returns the same instance (singleton)
      // const apollo = this.apolloClient;

      if (!process.browser) {
        try {
          // Run all GraphQL queries
          await getDataFromTree(
            <App
              {...appProps}
              Component={Component}
              router={router}
              apolloClient={apollo}
            />
          );
        } catch (error) {
          // Prevent Apollo Client GraphQL errors from crashing SSR.
          // Handle them in components via the data.error prop:
          // https://www.apollographql.com/docs/react/api/react-apollo.html#graphql-query-data-error
          console.error('Error while running `getDataFromTree`', error);
        }

        // getDataFromTree does not call componentWillUnmount
        // head side effect therefore need to be cleared manually
        Head.rewind();
      }

      // Extract query data from the Apollo store
      const apolloState = apollo.cache.extract();

      // TODO: Should we use accessToken here?
      // This returns props that are consumed by the constructor
      return {
        ...appProps,
        apolloState,
        accessToken
      };
    }

    constructor(props) {
      super(props);

      this.apolloClient = initApollo(props.apolloState, apolloConfig);
    }

    render() {
      return <App {...this.props} apolloClient={this.apolloClient} />;
    }
  };
};

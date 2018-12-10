// TODO: actually use user's github authenticated session
// TODO: json-pretty dep only for illustrative purposes, remove
// likely by proxy
import { Component, Fragment } from 'react';
import PropTypes from 'prop-types';
import getConfig from 'next/config';

import { ApolloClient } from 'apollo-client';
import { ApolloLink } from 'apollo-link';
import { HttpLink } from 'apollo-link-http';
import { setContext } from 'apollo-link-context';
import { InMemoryCache } from 'apollo-cache-inmemory';
import gql from 'graphql-tag';
import { Query } from 'react-apollo';

// For demonstrative purposes only
// TODO: Remove
import JSONPretty from 'react-json-pretty';

import ErrorMessage from '../components/ErrorMessage';

// TODO: modify initApollo to allow creation of multiple singleotn endpoints
// and use that along with withApolloClient
// serverRuntimeConfig is only seen during server renderes
const {
  serverRuntimeConfig,
  publicRuntimeConfig: { AUTH0 }
} = getConfig();

const httpLink = new HttpLink({
  uri: 'https://api.github.com/graphql' // Server URL (must be absolute)
});

const authLink = setContext((request, previousContext) => {
  // TODO: handle missing auth token
  return {
    headers: {
      ...previousContext.headers,
      authorization: `Bearer ${
        serverRuntimeConfig && serverRuntimeConfig.GITHUB
          ? serverRuntimeConfig.GITHUB.ACCESS_TOKEN
          : ''
      }`
    }
  };
});

let count = 0;
const cache = new InMemoryCache();

// TODO: reuse, properly taking into account browser vs server side rendering mode
const customClient = new ApolloClient({
  ssrMode: true,
  connectToDevTools: process.browser,
  link: ApolloLink.from([authLink, httpLink]),
  cache
});

const GITHUB_QUERY = gql`
  query {
    repository(owner: "hail-is", name: "hail") {
      issues(
        last: 100
        states: [OPEN]
        orderBy: { field: CREATED_AT, direction: ASC }
      ) {
        edges {
          node {
            ... on Issue {
              title

              assignees(last: 5) {
                edges {
                  node {
                    login
                  }
                }
              }

              labels(last: 5) {
                edges {
                  node {
                    name
                  }
                }
              }
            }
          }
        }
      }
    }
  }
`;

// TODO: Finish
class Scorecard extends Component {
  // Anything you want run on the server
  static async getInitialProps(ctx) {
    // Server
    if (ctx.req) {
    }

    return {};
  }

  componentDidMount = () => {};

  render() {
    const { classes } = this.props;

    return (
      <Query client={customClient} query={GITHUB_QUERY}>
        {({ loading, error, data }) => {
          // Just a javascript arrow function body...

          if (loading) return <div>Loading</div>;
          if (error) return <ErrorMessage error={error} />;

          console.log('data', data);

          return <div>Got data</div>;
        }}
      </Query>
    );
  }
}

export default Scorecard;

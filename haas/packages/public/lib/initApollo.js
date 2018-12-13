// TODO: Implement subscriptions
// TODO: create class abstraction, of which this will be a
// single, singleton implementation of
import { ApolloClient } from 'apollo-client';
import { HttpLink } from 'apollo-link-http';
import { setContext } from 'apollo-link-context';
import { InMemoryCache } from 'apollo-cache-inmemory';
// import { WebSocketLink } from 'apollo-link-ws';
// import { getMainDefinition } from 'apollo-utilities';

import { ApolloLink } from 'apollo-link';

import fetch from 'isomorphic-unfetch';

import getConfig from 'next/config';

const { publicRuntimeConfig } = getConfig();

// Beginning of websocket code
// const isSubscriptionOperation = ({ query }) => {
//   const { kind, operation } = getMainDefinition(query);
//   return kind === 'OperationDefinition' && operation === 'subscription';
// };

let apolloClient = null;

// Polyfill fetch() on the server (used by apollo-client)
if (!process.browser) {
  global.fetch = fetch;
}

function create(initialState, { initialAccessToken, getToken }) {
  // Beginning of websocket code
  // const wsLink = new WebSocketLink({
  //   uri: `ws://localhost:3000/graphql`,
  //   options: {
  //     reconnect: true
  //   }
  // });

  // const link = split(
  // split based on operation type
  // can just isSubscriptionOperation, but this is clearer to me
  //   ({ query }) => isSubscriptionOperation(query),
  //   wsLink,
  //   httpLink
  // );

  const httpLink = new HttpLink({
    uri: publicRuntimeConfig.GRAPHQL.ENDPOINT, // Server URL (must be absolute)
    credentials: 'same-origin' // Additional fetch() options like `credentials` or `headers`
  });

  const authLink = setContext((request, previousContext) => {
    // Do this because we may not yet be logged in, but may
    // store a token in a server-accessible way and if so may pass it in
    // the initial request;
    const token = initialAccessToken || getToken();

    return {
      headers: {
        ...previousContext.headers,
        authorization: token ? `Bearer ${token}` : ''
      }
    };
  });

  const link = ApolloLink.from([authLink, httpLink]);

  // Check out https://github.com/zeit/next.js/pull/4611 if you want to use the AWSAppSyncClient
  return new ApolloClient({
    connectToDevTools: process.browser,
    ssrMode: !process.browser, // Disables forceFetch on the server (so queries are only run once)
    // they have defined a method called "concat" on Apollo Link objects
    // this has nothing to do with Javascript's Array.concat
    // https://github.com/apollographql/apollo-link/blob/17b1fe6e30c720bfa7a0fed8fc62ebb48168758f/packages/apollo-link/src/link.ts
    link,
    cache: new InMemoryCache().restore(initialState || {})
  });
}

export default function initApollo(initialState, args) {
  // Make sure to create a new client for every server-side request so that data
  // isn't shared between connections (which would be bad)
  if (!process.browser) {
    return create(initialState, args);
  }

  // Reuse client on the client-side
  if (!apolloClient) {
    apolloClient = create(initialState, args);
  }

  return apolloClient;
}

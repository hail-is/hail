const { setContext } = require('apollo-link-context');
const { HttpLink } = require('apollo-link-http');
const fetch = require('isomorphic-unfetch');
const {
  introspectSchema,
  makeRemoteExecutableSchema
} = require('graphql-tools');

// console.info('user', user.middleware);
const http = new HttpLink({ uri: 'https://api.github.com/graphql', fetch });

// This is a public-only permissions token
// Used to read the schema before a user tries to access
// TODO: If not using async/await, doesn't work (promise only)
// not sure why, follow up
const tokenForScheamIntrospection =
  process.env.GITHUB_PERSONAL_TOKEN_PUBLIC_ONLY;

module.exports = async user => {
  const { middleware } = user;
  let isFirst = true;

  const link = setContext(async (_, ctx) => {
    if (isFirst) {
      isFirst = false;
      return {
        headers: {
          Authorization: `bearer ${tokenForScheamIntrospection}`
        }
      };
    }

    // The personal access token that has public-only permissions
    // is a fallback, in case 1) no user and the query requires
    // only public permission
    // 2) A user, but not logged in with github, and doesn't need private permissions
    // In the case private permission are requested, a 401 should be issued
    // TODO: Write tests for this
    if (!(ctx && ctx.graphqlContext && ctx.graphqlContext.user)) {
      return {
        headers: {
          Authorization: `bearer ${tokenForScheamIntrospection}`
        }
      };
    }

    try {
      const token = await middleware.extractAccessToken(
        ctx.graphqlContext.user
      );

      return {
        headers: {
          Authorization: `bearer ${token}`
        }
      };
    } catch (e) {
      return {
        headers: {
          Authorization: `bearer ${tokenForScheamIntrospection}`
        }
      };
    }
  }).concat(http);

  const schema = await introspectSchema(link);

  const executableSchema = makeRemoteExecutableSchema({
    schema,
    link
  });

  return executableSchema;
};

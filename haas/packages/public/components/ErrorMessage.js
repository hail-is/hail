import { LoginLink } from './Header';

export default ({ error }) => {
  const { graphQLErrors, networkError } = error;

  console.info('ERROR', graphQLErrors, networkError, error.message);
  let isAuthError;
  // let error;
  if (graphQLErrors) {
    isAuthError =
      graphQLErrors.filter(e => e.code === 'UNAUTHENTICATED').length > 0;
  }

  console.info('status', isAuthError);
  if (isAuthError) {
    return <LoginLink />;
  }

  return <aside className="error">{error.message}</aside>;
};

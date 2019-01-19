const polka = require('polka');

const PORT = 8000;
// Polka is *mostly* express compatible, but let's not use unnecessary libraries
// better to understand the underlying security issues
// Best practices state that we should check the exact form of the auth header
// and if accepting from more than 1 place (say body or get query)
// make sure the token exists in only one place
// We have no need for this. Accept only from the header

// Also, I prefer not using middleware; call the functions needed inside the route
const bearerPrefix = 'Bearer ';
// naturally 1 past; so useful as end, and as the index to start reading token from
const bearerPrefixLen = bearerPrefix.length;
const getAuthToken = req => {
  // This is set from an "Authorization" header
  if (req.headers.authorization) {
    console.info('header auth', req.headers.authorization);

    if (req.headers.authorization.substr(0, bearerPrefixLen) !== bearerPrefix) {
      throw new Error('WTF');
    }

    return req.headers.authorization.substr(bearerPrefixLen);
  }

  return null;
};

polka()
  .get('/notebook', (req, res) => {
    const token = getAuthToken(req);
  })
  .listen(PORT, err => {
    if (err) {
      throw err;
    }

    console.info(`Auth server running on port ${PORT}`);
  });

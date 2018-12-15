module.exports = (userInstance /*config*/) => {
  const uM = userInstance.middleware;

  const route1 = {
    method: 'GET',
    path: '/api/ci',
    handlers: [
      uM.verifyToken,
      uM.getAuth0ProviderAccessToken,
      async (req, res) => {
        console.info(req.accessToken);
        res.end();
      }
    ]
  };

  return {
    routes: [route1]
  };
};

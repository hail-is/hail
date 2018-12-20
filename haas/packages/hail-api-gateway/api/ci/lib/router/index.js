// Its job wil lbe to interact with the CI interface in an authenticated way
// TODO: specify API;
// I propose something like: /api/ci?q=/ci/endpoint
module.exports = userInstance => {
  const uM = userInstance.middleware;

  const route1 = {
    method: 'GET',
    path: '/ci',
    handlers: [
      uM.verifyToken,
      uM.getAuth0ProviderAccessToken,
      async (req, res) => {
        res.end();
      }
    ]
  };

  return {
    routes: [route1]
  };
};

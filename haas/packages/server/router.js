module.exports = function appRouter(app, routes, config) {
  const publicPath = config && config.router.publicPath;
  const spaDelim = config && config.spaDelimeter || "#!";
  if (!publicPath) {
    throw new Error('Must provide a config with a publicPath property');
  }

  app.get('/api/ping', (req, res) => {
    res.sendStatus(200);
  });

  routes.forEach((route) => {
    app.use(`/api/${route.routesRootName}`, route.router );
  });

  // Everything else goes to the single page application router
  app.route('/*').all((req, res) => {
    // console.info('got all', req.originalUrl);
  //   // const fullUrl = req.protocol + '://' + req.get('host');
  //   // // const urlPath = req.originalUrl.replace(/\//, spaDelim);
  //   // console.info(`${fullUrl}/${req.originalUrl}`);
  //   // res.redirect(`${fullUrl}/${req.originalUrl}`);
    res.sendFile(publicPath + '/index.html');
  });
};

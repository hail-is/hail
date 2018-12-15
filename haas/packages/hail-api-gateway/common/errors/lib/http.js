module.exports = (function httpErrors() {
  return { forbidden };

  function forbidden(req, res) {
    res.sendStatus(403);
  }
})();

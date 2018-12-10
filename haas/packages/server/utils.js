/**
 * Combine multiple middleware together.
 *
 * @param {Function[]} mids functions of form:
 *   function(req, res, next) { ... }
 * @return {Function} single combined middleware
 */
module.exports.combineMiddleware = function combineMiddleware(list) {
  return function (req, res, next) {
    (function iter(i) {
      var mid = list[i]
      if (!mid) return next()
      mid(req, res, function (err) {
        if (err) return next(err)
        iter(i+1)
      })
    }(0))
  }
}
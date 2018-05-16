package is.hail.expr.ir

object Binds {
  def apply(x: IR): Set[String] = {
    x match {
      case Let(n, _, _) => Set(n)
      case ArrayMap(_, n, _) => Set(n)
      case ArrayFlatMap(_, n, _) => Set(n)
      case ArrayFilter(_, n, _) => Set(n)
      case ArrayFold(_, _, accumName, valueName, _) => Set(accumName, valueName)
      case _ => Set.empty[String]
    }
  }
}

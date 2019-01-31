package is.hail.expr.ir

object Binds {
  def apply(x: IR, v: String, i: Int): Boolean = {
    x match {
      case Let(n, _, _) =>
        v == n && i == 1
      case ArrayMap(_, n, _) =>
        v == n && i == 1
      case ArrayFlatMap(_, n, _) =>
        v == n && i == 1
      case ArrayFilter(_, n, _) =>
        v == n && i == 1
      case ArrayFold(_, _, accumName, valueName, _) =>
        (v == accumName || v == valueName) && i == 2
      case ArrayScan(_, _, accumName, valueName, _) =>
        (v == accumName || v == valueName) && i == 2
      case ArrayLeftJoinDistinct(_, _, l, r, _, _) =>
        (v == l || v == r) && i == 2
      case AggExplode(_, n, _) =>
        v == n && i == 1
      case _ =>
        false
    }
  }
}

object Bindings {
  private val empty = Array[String]()

  def apply(x: IR): Array[String] = x match {
    case Let(name, _, _) => Array(name)
    case ArrayMap(_, name, _) => Array(name)
    case ArrayFlatMap(_, name, _) => Array(name)
    case ArrayFilter(_, name, _) => Array(name)
    case ArrayFold(_, _, accumName, valueName, _) => Array(accumName, valueName)
    case ArrayScan(_, _, accumName, valueName, _) => Array(accumName, valueName)
    case ArrayLeftJoinDistinct(_, _, l, r, _, _) => Array(l, r)
    case AggExplode(_, n, _) => Array(n)
    case _ => empty
  }
}

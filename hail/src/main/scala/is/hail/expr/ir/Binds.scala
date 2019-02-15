package is.hail.expr.ir

object Binds {
  def apply(x: IR, v: String, i: Int): Boolean = {
    x match {
      case Let(n, _, _) =>
        v == n && i == 1
      case AggLet(n, _, _) =>
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
      case ArraySort(_, l, r, _) =>
        (v == l || v == r) && i == 2
      case CollectDistributedArray(_, _, n1, n2, _) =>
        (v == n1 || v == n2) && i == 2
      case _ =>
        false
    }
  }
}

object AggBinds {
  def apply(x: IR, v: String, i: Int): Boolean = {
    x match {
      case AggLet(n, _, _) =>
        v == n && i == 1
      case AggExplode(_, n, _) =>
        v == n && i == 1
      case AggArrayPerElement(_, n, _) =>
        v == n && i == 1
      case ArrayAgg(_, name, _) =>
        v == name && i == 1
      case _ =>
        false
    }
  }
}


object Bindings {
  val empty: Array[String] = Array[String]()

  def apply(x: IR): Array[String] = x match {
    case Let(name, _, _) => Array(name)
    case ArrayMap(_, name, _) => Array(name)
    case ArrayFor(_, name, _) => Array(name)
    case ArrayFlatMap(_, name, _) => Array(name)
    case ArrayFilter(_, name, _) => Array(name)
    case ArrayFold(_, _, accumName, valueName, _) => Array(accumName, valueName)
    case ArrayScan(_, _, accumName, valueName, _) => Array(accumName, valueName)
    case ArrayLeftJoinDistinct(_, _, l, r, _, _) => Array(l, r)
    case ArraySort(_, left, right, _) => Array(left, right)
    case CollectDistributedArray(_, _, cname, gname, _) => Array(cname, gname)
    case _ => empty
  }
}

object AggBindings {
  def apply(x: IR): Array[String] = x match {
    case AggLet(name, _, _) => Array(name)
    case AggExplode(_, name, _) => Array(name)
    case ArrayAgg(_, name, _) => Array(name)
    case AggArrayPerElement(_, name, _) => Array(name)
    case _ => Bindings.empty
  }
}
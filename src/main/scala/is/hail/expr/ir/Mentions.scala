package is.hail.expr.ir

object Mentions {
  def apply(x: IR, v: String): Boolean = {
    x match {
      case Ref(n, _) => v == n
      case _ =>
        if (Binds(x).contains(v))
          false
        else
          Children(x).exists {
            case c: IR => Mentions(c, v)
            case _ => false
          }
    }
  }
}

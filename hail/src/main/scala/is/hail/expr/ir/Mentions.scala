package is.hail.expr.ir

object Mentions {
  def apply(x: IR, v: String): Boolean = FreeVariables(x).exists(_.name == v)
}
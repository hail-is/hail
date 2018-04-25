package is.hail.expr.ir

object Recur {
  def apply(f: IR => IR)(ir: IR): IR = Copy(ir, Children(ir).map {
    case c: IR => f(c)
    case c => c
  }).asInstanceOf[IR]
}

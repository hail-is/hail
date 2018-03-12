package is.hail.expr.ir

object Optimize {

  def apply(ir: IR): IR = {
    Recur(opt)(ir)
  }

  def opt(ir: IR): IR = ir match {
    case If(ApplyUnaryPrimOp(Bang(), IsNA(x), _), y, NA(_), _) => y
    case _ => ir
  }

}

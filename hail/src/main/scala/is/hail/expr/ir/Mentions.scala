package is.hail.expr.ir

object Mentions {
  def apply(x: IR, v: Name): Boolean = {
    val fv = FreeVariables(x, false, false)
    fv.eval.lookupOption(v).isDefined
  }

  def inAggOrScan(x: IR, name: Name): Boolean = {
    val fv = FreeVariables(x, true, true)
    fv.agg.get.lookupOption(name).isDefined || fv.scan.get.lookupOption(name).isDefined
  }
}

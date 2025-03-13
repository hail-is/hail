package is.hail.expr.ir

object IRSize {
  def apply(ir0: BaseIR): Int = {

    var size = 0

    def visit(ir: BaseIR): Unit = {
      size += 1
      ir.children.foreach(visit)
    }

    visit(ir0)
    size
  }
}

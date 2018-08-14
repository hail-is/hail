package is.hail.expr.ir

object MapIR {
  def apply(f: IR => IR)(ir: IR): IR = Copy(ir, Children(ir).map {
    case c: IR => f(c)
    case c => c
  }).asInstanceOf[IR]

  def mapBaseIR(ir: BaseIR, f: BaseIR => BaseIR): BaseIR = f(ir.copy(newChildren = ir.children.map(mapBaseIR(_, f))))
}

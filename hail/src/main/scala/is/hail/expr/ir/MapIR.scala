package is.hail.expr.ir

object MapIR {
  def apply(f: IR => IR)(ir: IR): IR = ir match {
    case ta: TableAggregate => ta
    case ma: MatrixAggregate => ma
    case _ => Copy(ir, ir.childrenSeq.map {
      case c: IR => f(c)
      case c => c
    })
  }

  def mapBaseIR(ir: BaseIR, f: BaseIR => BaseIR): BaseIR = f(ir.mapChildren(mapBaseIR(_, f)))
}

object VisitIR {
  def apply(ir: BaseIR)(f: BaseIR => Unit): Unit = {
    f(ir)
    ir.children.foreach(apply(_)(f))
  }
}
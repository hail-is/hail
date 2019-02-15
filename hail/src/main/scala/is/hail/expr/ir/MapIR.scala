package is.hail.expr.ir

object VisitIR {
  def apply(f: IR => Unit)(ir: IR): Unit = {
    Children(ir).foreach {
      case c: IR => f(c)
      case c => c
    }
  }
}

object MapIR {
  def apply(f: IR => IR)(ir: IR): IR = Copy(ir, Children(ir).map {
    case c: IR => f(c)
    case c => c
  }).asInstanceOf[IR]

  def mapBaseIR(ir: BaseIR, f: BaseIR => BaseIR): BaseIR = f(ir.copy(newChildren = ir.children.map(mapBaseIR(_, f))))
}

/**
  * Would be more elegant in a mutable IR with parent pointers
  */
object MapIRSubtrees {
  def apply(f: IR => IR)(ir: BaseIR): BaseIR = {
    RewriteBottomUp(ir, {
      case _: IR => None
      case _ => Some(ir.copy(ir.children.map {
        case child: IR => f(child)
        case c => c
      }))
    }) match {
      case ir: IR => f(ir)
      case x => x
    }
  }
}
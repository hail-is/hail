package is.hail.expr.ir

object VisitIRChildren {
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
  def apply(f: IR => IR)(ir0: BaseIR): BaseIR = {
    def rewrite(ir: BaseIR): BaseIR = {
      ir match {
        case valueIR: IR => valueIR.copy(valueIR.children.map(rewrite))
        case relational => relational.copy(relational.children.map {
          rewrite(_) match {
            case child: IR => f(child)
            case child => child
          }
        })
      }
    }

    rewrite(ir0) match {
      case ir: IR => f(ir)
      case x => x
    }
  }
}
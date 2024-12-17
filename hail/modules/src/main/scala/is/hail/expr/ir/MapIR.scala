package is.hail.expr.ir

import is.hail.types.virtual.Type

object MapIR {
  def apply(f: IR => IR)(ir: IR): IR = ir match {
    case ta: TableAggregate => ta
    case ma: MatrixAggregate => ma
    case _ => ir.mapChildren {
        case c: IR => f(c)
        case c => c
      }
  }

  def mapBaseIR(ir: BaseIR, f: BaseIR => BaseIR): BaseIR = f(ir.mapChildren(mapBaseIR(_, f)))
}

object VisitIR {
  def apply(ir: BaseIR)(f: BaseIR => Unit): Unit = {
    f(ir)
    ir.children.foreach(apply(_)(f))
  }

  def withEnv[V, E <: GenericBindingEnv[E, Type]](ir: BaseIR, env: E)(f: (BaseIR, E) => Unit)
    : Unit = {
    f(ir, env)
    ir.forEachChildWithEnv(env)(withEnv(_, _)(f))
  }
}

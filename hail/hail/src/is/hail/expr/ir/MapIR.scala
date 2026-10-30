package is.hail.expr.ir

import is.hail.expr.ir.defs.{MatrixAggregate, TableAggregate}
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

  def withEnv[E[_]: Scopes, A](
    ir: BaseIR,
    env: E[A],
  )(
    v: (Name, Type) => A
  )(
    f: (BaseIR, E[A]) => Unit
  ): Unit = {
    f(ir, env)
    ir.forEachChildWithEnv(env)(v)(withEnv(_, _)(v)(f))
  }
}

package is.hail.expr.ir.lowering

import is.hail.backend.ExecuteContext
import is.hail.expr.ir.{
  BaseIR, BlockMatrixIR, Compilable, Emittable, IR, IRTraversal, MatrixIR, RelationalLetMatrixTable,
  RelationalLetTable, TableIR, TableKeyBy, TableKeyByAndAggregate, TableOrderBy,
}
import is.hail.expr.ir.defs.{ApplyIR, RelationalLet, RelationalRef}
import is.hail.utils.implicits.toRichPredicate

abstract class Invariant(implicit E: sourcecode.Enclosing) extends (BaseIR => Boolean) {
  final def verify(ctx: ExecuteContext, ir: BaseIR): Unit =
    ctx.time {
      IRTraversal.levelOrder(ir).foreach { ir =>
        if (!apply(ir))
          throw new RuntimeException(
            s"lowered state ${this.getClass.getCanonicalName} forbids IR $ir"
          )
      }
    }
}

object Invariant {

  implicit def apply(p: BaseIR => Boolean)(implicit E: sourcecode.Enclosing): Invariant =
    new Invariant() { override def apply(ir: BaseIR): Boolean = p(ir) }

  lazy val AnyIR: Invariant =
    Invariant(_ => true)

  lazy val NoMatrixIR: Invariant =
    Invariant(!_.isInstanceOf[MatrixIR])

  lazy val NoTableIR: Invariant =
    Invariant(!_.isInstanceOf[TableIR])

  lazy val NoBlockMatrixIR: Invariant =
    Invariant(!_.isInstanceOf[BlockMatrixIR])

  lazy val NoRelationalLets: Invariant =
    Invariant {
      case _: RelationalLet => false
      case _: RelationalLetMatrixTable => false
      case _: RelationalLetTable => false
      case _: RelationalRef => false
      case _ => true
    }

  lazy val NoApplyIR: Invariant =
    Invariant(!_.isInstanceOf[ApplyIR])

  lazy val ValueIROnly: Invariant =
    Invariant(_.isInstanceOf[IR])

  lazy val CompilableValueIRs: Invariant =
    Invariant {
      case x: IR => Compilable(x)
      case _ => true
    }

  lazy val CompilableIR: Invariant =
    ValueIROnly and CompilableValueIRs

  lazy val EmittableValueIRs: Invariant =
    Invariant {
      case x: IR => Emittable(x)
      case _ => true
    }

  lazy val EmittableIR: Invariant =
    ValueIROnly and EmittableValueIRs

  lazy val LoweredShuffles: Invariant =
    Invariant {
      case t: TableKeyBy => t.definitelyDoesNotShuffle
      case _: TableKeyByAndAggregate => false
      case t: TableOrderBy => t.definitelyDoesNotShuffle
      case _ => true
    }
}

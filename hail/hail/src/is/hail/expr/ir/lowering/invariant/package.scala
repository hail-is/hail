package is.hail.expr.ir.lowering

import is.hail.backend.ExecuteContext
import is.hail.expr.ir.{
  BaseIR, BlockMatrixIR, Compilable, Emittable, IR, IRTraversal, MatrixIR, NormalizeNames, Pretty,
  RelationalLetMatrixTable, RelationalLetTable, TableIR, TableKeyBy, TableKeyByAndAggregate,
  TableOrderBy,
}
import is.hail.expr.ir.defs.{ApplyIR, LiftMeOut, RelationalLet, RelationalRef}
import is.hail.expr.ir.lowering.invariant.implicits.RichInvariantOps
import is.hail.utils.fatal
import is.hail.utils.implicits.toRichPredicate

package invariant {
  sealed abstract class Invariant {
    def verify(ctx: ExecuteContext, ir: BaseIR): Unit
  }

  case object NoOp extends Invariant {
    override def verify(ctx: ExecuteContext, ir: BaseIR): Unit = ()
  }

  abstract class Global(implicit E: sourcecode.Enclosing) extends Invariant {

    def local(ir: BaseIR): Boolean

    final override def verify(ctx: ExecuteContext, ir: BaseIR): Unit =
      ctx.time {
        IRTraversal.levelOrder(ir).foreach { ir =>
          if (!local(ir)) fatal(
            s"Invariant ${E.value} forbids IR ${Pretty(ctx, ir, preserveNames = true)}"
          )
        }
      }
  }
}

package object invariant {
  object implicits {
    implicit class RichInvariantOps(private val x: Invariant) extends AnyVal {
      def and(y: Invariant)(implicit E: sourcecode.Enclosing): Invariant =
        x match {
          case NoOp => y
          case p: Global => y match {
              case NoOp => p
              case q: Global => Invariant(p.local _ and q.local)
            }
        }
    }
  }

  def Invariant(p: BaseIR => Boolean)(implicit E: sourcecode.Enclosing): Invariant =
    new Global { override def local(ir: BaseIR): Boolean = p(ir) }

  def NameNormalizedIr: Invariant =
    if (is.hail.StrictLoweringInvariants) new NormalizeNames.Invariant
    else NoOp

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

  lazy val NoTableKeyByAndAggregate: Invariant =
    Invariant(!_.isInstanceOf[TableKeyByAndAggregate])

  lazy val NoLiftMeOuts: Invariant =
    Invariant(!_.isInstanceOf[LiftMeOut])

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

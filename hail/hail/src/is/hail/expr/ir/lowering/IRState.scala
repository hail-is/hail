package is.hail.expr.ir.lowering

import is.hail.backend.ExecuteContext
import is.hail.expr.ir.{BaseIR, IRTraversal, TableKeyBy, TableKeyByAndAggregate, TableOrderBy}
import is.hail.expr.ir.defs.{RelationalLet, RelationalRef}
import is.hail.expr.ir.lowering.Rules._
import is.hail.utils.toRichPredicate

abstract class IRState(implicit E: sourcecode.Enclosing) {
  protected def allows: Rule

  final def verify(ctx: ExecuteContext, ir: BaseIR): Unit =
    ctx.time {
      IRTraversal.levelOrder(ir).foreach { ir =>
        if (!allows(ir))
          throw new RuntimeException(
            s"lowered state ${this.getClass.getCanonicalName} forbids IR $ir"
          )
      }
    }

  def +(other: IRState)(implicit E: sourcecode.Enclosing): IRState =
    new IRState()(E) {
      override val allows: Rule = allows and other.allows
    }
}

case object AnyIR extends IRState {
  override val allows: Rule = _ => true
}

case object MatrixLoweredToTable extends IRState {
  override val allows: Rule = NoMatrixIR
}

case object ExecutableTableIR extends IRState {
  override val allows: Rule = NoMatrixIR and NoRelationalLets and CompilableValueIRs
}

case object CompilableIR extends IRState {
  override val allows: Rule = ValueIROnly and CompilableValueIRs
}

case object CompilableIRNoApply extends IRState {
  override val allows: Rule = ValueIROnly and CompilableValueIRs and NoApplyIR
}

case object EmittableIR extends IRState {
  override val allows: Rule = ValueIROnly and EmittableValueIRs
}

case object EmittableStreamIRs extends IRState {
  override val allows: Rule = ValueIROnly and EmittableValueIRs
}

case object NoRelationalLetsState extends IRState {
  override val allows: Rule = {
    case _: RelationalRef => false
    case _: RelationalLet => false
    case _ => true
  }
}

case object LoweredShuffles extends IRState {
  override val allows: Rule = {
    case t: TableKeyBy => t.definitelyDoesNotShuffle
    case _: TableKeyByAndAggregate => false
    case t: TableOrderBy => t.definitelyDoesNotShuffle
    case _ => true
  }
}

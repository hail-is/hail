package is.hail.expr.ir.lowering

import is.hail.backend.ExecuteContext
import is.hail.expr.ir.{
  BaseIR, IRTraversal, RelationalLet, RelationalRef, TableKeyBy, TableKeyByAndAggregate,
  TableOrderBy,
}

abstract class IRState(implicit E: sourcecode.Enclosing) {
  protected val rules: Array[Rule]

  final def verify(ctx: ExecuteContext, ir: BaseIR): Unit =
    ctx.time {
      IRTraversal.levelOrder(ir).foreach { ir =>
        if (!rules.forall(_.allows(ir)))
          throw new RuntimeException(
            s"lowered state ${this.getClass.getCanonicalName} forbids IR $ir"
          )
      }
    }

  def +(other: IRState)(implicit E: sourcecode.Enclosing): IRState = {
    val newRules = rules ++ other.rules
    new IRState()(E) {
      val rules: Array[Rule] = newRules
    }
  }
}

case object AnyIR extends IRState {
  val rules: Array[Rule] = Array()
}

case object MatrixLoweredToTable extends IRState {
  val rules: Array[Rule] = Array(NoMatrixIR)
}

case object ExecutableTableIR extends IRState {
  val rules: Array[Rule] = Array(NoMatrixIR, NoRelationalLets, CompilableValueIRs)
}

case object CompilableIR extends IRState {
  val rules: Array[Rule] = Array(ValueIROnly, CompilableValueIRs)
}

case object CompilableIRNoApply extends IRState {
  val rules: Array[Rule] = Array(ValueIROnly, CompilableValueIRs, NoApplyIR)
}

case object EmittableIR extends IRState {
  val rules: Array[Rule] = Array(ValueIROnly, EmittableValueIRs)
}

case object EmittableStreamIRs extends IRState {
  val rules: Array[Rule] = Array(ValueIROnly, EmittableValueIRs)
}

case object NoRelationalLetsState extends IRState {
  val rules: Array[Rule] = Array {
    case _: RelationalRef => false
    case _: RelationalLet => false
    case _ => true
  }
}

case object LoweredShuffles extends IRState {
  val rules: Array[Rule] = Array {
    case t: TableKeyBy => t.definitelyDoesNotShuffle
    case _: TableKeyByAndAggregate => false
    case t: TableOrderBy => t.definitelyDoesNotShuffle
    case _ => true
  }
}

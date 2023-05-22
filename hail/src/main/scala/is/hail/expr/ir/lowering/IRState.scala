package is.hail.expr.ir.lowering

import is.hail.expr.ir.lowering.Rule.Implicits._
import is.hail.expr.ir.{BaseIR, IRTraversal, RelationalLet, RelationalRef, TableKeyBy, TableKeyByAndAggregate, TableOrderBy}

trait IRState {

  type AnyRule = Rule[Any]
  val rules: Array[AnyRule]

  def verify(ir: BaseIR): Unit =
    IRTraversal.preOrder(ir).foldLeft(rules.map(_.init)) { (state, ir) =>
      rules.zip(state).map { case (rule, state_) =>
        val (state, passes) = rule(state_, ir)
        if (!passes) throw new RuntimeException(s"lowered state ${ this.getClass.getCanonicalName } forbids IR $ir")
        state
      }
    }

  def +(other: IRState): IRState = {
    val newRules = rules ++ other.rules
    new IRState {
      val rules: Array[AnyRule] = newRules
    }
  }
}

case object AnyIR extends IRState {
  val rules: Array[AnyRule] = Array()
}

case object MatrixLoweredToTable extends IRState {
  val rules: Array[AnyRule] =
    Array(NoMatrixIR)
}

case object ExecutableTableIR extends IRState {
  val rules: Array[AnyRule] =
    Array(NoMatrixIR, NoRelationalLets, CompilableValueIRs)
}

case object CompilableIR extends IRState {
  val rules: Array[AnyRule] =
    Array(ValueIROnly, CompilableValueIRs)
}

case object CompilableIRNoApply extends IRState {
  val rules: Array[AnyRule] =
    Array(ValueIROnly, CompilableValueIRs, NoApplyIR)
}

case object EmittableIR extends IRState {
  val rules: Array[AnyRule] =
    Array(ValueIROnly, EmittableValueIRs)
}

case object EmittableStreamIRs extends IRState {
  val rules: Array[AnyRule] =
    Array(ValueIROnly, EmittableValueIRs)
}

case object NoRelationalLetsState extends IRState {
  val rules: Array[AnyRule] = Array(
    PureRule({
      case _: RelationalRef => false
      case _: RelationalLet => false
      case _ => true
    })
  )
}

case object LoweredShuffles extends IRState {
  val rules: Array[AnyRule] = Array(
    PureRule({
      case t: TableKeyBy => t.definitelyDoesNotShuffle
      case _: TableKeyByAndAggregate => false
      case t: TableOrderBy => t.definitelyDoesNotShuffle
      case _ => true
    })
  )
}

case object RootSemanticHash extends IRState {
  override val rules: Array[AnyRule] =
    Array(HasRootSemanticHash)

}
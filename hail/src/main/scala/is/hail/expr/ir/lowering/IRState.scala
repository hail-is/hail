package is.hail.expr.ir.lowering

import is.hail.expr.ir.{
  BaseIR, RelationalLet, RelationalRef, TableKeyBy, TableKeyByAndAggregate, TableOrderBy,
}

trait IRState {

  val rules: Array[Rule]

  final def allows(ir: BaseIR): Boolean = rules.forall(_.allows(ir))

  final def verify(ir: BaseIR): Unit = {
    if (!rules.forall(_.allows(ir)))
      throw new RuntimeException(s"lowered state ${this.getClass.getCanonicalName} forbids IR $ir")
    ir.children.foreach(verify)
  }

  final def permits(ir: BaseIR): Boolean = rules.forall(_.allows(ir)) && ir.children.forall(permits)

  def +(other: IRState): IRState = {
    val newRules = rules ++ other.rules
    new IRState {
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
  val rules: Array[Rule] = Array(
    new Rule {
      def allows(ir: BaseIR): Boolean = ir match {
        case _: RelationalRef => false
        case _: RelationalLet => false
        case _ => true
      }
    }
  )
}

case object LoweredShuffles extends IRState {
  val rules: Array[Rule] = Array(
    new Rule {
      def allows(ir: BaseIR): Boolean = ir match {
        case t: TableKeyBy => t.definitelyDoesNotShuffle
        case _: TableKeyByAndAggregate => false
        case t: TableOrderBy => t.definitelyDoesNotShuffle
        case _ => true
      }
    }
  )
}

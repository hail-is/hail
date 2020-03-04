package is.hail.expr.ir.lowering

import is.hail.expr.ir.BaseIR

trait IRState {

  val rules: Array[Rule]

  final def allows(ir: BaseIR): Boolean = rules.forall(_.allows(ir))

  final def verify(ir: BaseIR): Unit = {
    if (!rules.forall(_.allows(ir)))
      throw new RuntimeException(s"lowered state ${ this.getClass.getCanonicalName } forbids IR $ir")
    ir.children.foreach(verify)
  }

  final def permits(ir: BaseIR): Boolean = rules.forall(_.allows(ir)) && ir.children.forall(permits)
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

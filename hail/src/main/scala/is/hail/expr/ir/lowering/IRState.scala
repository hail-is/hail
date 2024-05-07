package is.hail.expr.ir.lowering

import is.hail.expr.ir.{
  BaseIR, RelationalLet, RelationalLetBlockMatrix, RelationalLetMatrixTable, RelationalLetTable,
}

import scala.annotation.tailrec

trait IRState {
  def verify(ir: BaseIR): Unit

  def permits(ir: BaseIR): Boolean
}

trait SimpleIRState extends IRState {

  protected val rules: Array[Rule]

  final def allows(ir: BaseIR): Boolean = rules.forall(_.allows(ir))

  final def verify(ir: BaseIR): Unit = {
    if (!rules.forall(_.allows(ir)))
      throw new RuntimeException(s"lowered state ${this.getClass.getCanonicalName} forbids IR $ir")
    ir.children.foreach(verify)
  }

  final def permits(ir: BaseIR): Boolean = rules.forall(_.allows(ir)) && ir.children.forall(permits)

  def +(other: SimpleIRState): SimpleIRState = {
    val newRules = rules ++ other.rules
    new SimpleIRState {
      val rules: Array[Rule] = newRules
    }
  }
}

final case class AllowTopLevelRelationalLets(nestedState: IRState) extends IRState {
  @tailrec override def verify(ir: BaseIR): Unit = ir match {
    case RelationalLet(_, value, body) =>
      nestedState.verify(value)
      verify(body)
    case RelationalLetBlockMatrix(_, value, body) =>
      nestedState.verify(value)
      verify(body)
    case RelationalLetMatrixTable(_, value, body) =>
      nestedState.verify(value)
      verify(body)
    case RelationalLetTable(_, value, body) =>
      nestedState.verify(value)
      verify(body)
    case ir => nestedState.verify(ir)
  }

  @tailrec override def permits(ir: BaseIR): Boolean = ir match {
    case RelationalLet(_, value, body) =>
      nestedState.permits(value)
      permits(body)
    case RelationalLetBlockMatrix(_, value, body) =>
      nestedState.permits(value)
      permits(body)
    case RelationalLetMatrixTable(_, value, body) =>
      nestedState.permits(value)
      permits(body)
    case RelationalLetTable(_, value, body) =>
      nestedState.permits(value)
      permits(body)
    case ir => nestedState.permits(ir)
  }
}

case object AnyIR extends SimpleIRState {
  val rules: Array[Rule] = Array()
}

case object NoRelationalLetsState extends SimpleIRState {
  val rules: Array[Rule] = Array(NoRelationalLets)
}

case object MatrixLoweredToTable extends SimpleIRState {
  val rules: Array[Rule] = Array(NoRelationalLets, NoMatrixIR)
}

case object LoweredShuffles extends SimpleIRState {
  val rules: Array[Rule] = Array(NoRelationalLets, NoMatrixIR, NoShuffles)
}

case object ExecutableTableIR extends SimpleIRState {
  val rules: Array[Rule] = Array(NoRelationalLets, NoMatrixIR, NoShuffles, CompilableValueIRs)
}

case object CompilableIR extends SimpleIRState {
  val rules: Array[Rule] = Array(ValueIROnly, CompilableValueIRs)
}

case object CompilableIRNoApply extends SimpleIRState {
  val rules: Array[Rule] = Array(ValueIROnly, CompilableValueIRs, NoApplyIR)
}

case object EmittableIR extends SimpleIRState {
  val rules: Array[Rule] = Array(ValueIROnly, EmittableValueIRs)
}

case object EmittableStreamIRs extends SimpleIRState {
  val rules: Array[Rule] = Array(ValueIROnly, EmittableValueIRs)
}

package is.hail.expr.ir

object InterpretableButNotCompilable {
  def apply(x: IR): Boolean = x match {
    case _: LiftMeOut => true
    case _: TableCount => true
    case _: TableGetGlobals => true
    case _: TableCollect => true
    case _: TableAggregate => true
    case _: MatrixAggregate => true
    case _: TableWrite => true
    case _: MatrixWrite => true
    case _: MatrixMultiWrite => true
    case _: TableMultiWrite => true
    case _: BlockMatrixWrite => true
    case _: BlockMatrixMultiWrite => true
    case _: TableToValueApply => true
    case _: MatrixToValueApply => true
    case _: BlockMatrixToValueApply => true
    case _: BlockMatrixCollect => true
    case _: BlockMatrixToTableApply => true
    case _ => false
  }
}

object Compilable {
  def apply(ir: IR): Boolean = {
    ir match {
      case _: LiftMeOut => false
      case _: TableCount => false
      case _: TableGetGlobals => false
      case _: TableCollect => false
      case _: TableAggregate => false
      case _: MatrixAggregate => false
      case _: TableWrite => false
      case _: MatrixWrite => false
      case _: MatrixMultiWrite => false
      case _: TableMultiWrite => false
      case _: BlockMatrixCollect => false
      case _: BlockMatrixWrite => false
      case _: BlockMatrixMultiWrite => false
      case _: TableToValueApply => false
      case _: MatrixToValueApply => false
      case _: BlockMatrixToValueApply => false
      case _: BlockMatrixToTableApply => false
      case _: RelationalRef => false
      case _: RelationalLet => false
      case _ => true
    }
  }
}

object Emittable {
  def isNonEmittableAgg(ir: IR): Boolean = ir match {
    case _: StreamAgg => true
    case _: StreamAggScan => true
    case _: ApplyAggOp => true
    case _: AggArrayPerElement => true
    case _: AggFilter => true
    case _: AggGroupBy => true
    case _: ApplyScanOp => true
    case _: AggExplode => true
    case _ => false
  }

  def apply(ir: IR): Boolean = ir match {
    case x if isNonEmittableAgg(x) => false
    case _: ApplyIR => false
    case x => Compilable(x)
  }
}

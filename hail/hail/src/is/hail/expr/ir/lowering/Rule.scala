package is.hail.expr.ir.lowering

import is.hail.expr.ir._

trait Rule {
  def allows(ir: BaseIR): Boolean
}

case object NoMatrixIR extends Rule {
  def allows(ir: BaseIR): Boolean = !ir.isInstanceOf[MatrixIR]
}

case object NoTableIR extends Rule {
  def allows(ir: BaseIR): Boolean = !ir.isInstanceOf[TableIR]
}

case object NoBlockMatrixIR extends Rule {
  def allows(ir: BaseIR): Boolean = !ir.isInstanceOf[BlockMatrixIR]
}

case object NoRelationalLets extends Rule {
  def allows(ir: BaseIR): Boolean = ir match {
    case _: RelationalLet => false
    case _: RelationalLetBlockMatrix => false
    case _: RelationalLetMatrixTable => false
    case _: RelationalLetTable => false
    case _: RelationalRef => false
    case _ => true
  }
}

case object CompilableValueIRs extends Rule {
  def allows(ir: BaseIR): Boolean = ir match {
    case x: IR => Compilable(x)
    case _ => true
  }
}

case object NoApplyIR extends Rule {
  override def allows(ir: BaseIR): Boolean = ir match {
    case _: ApplyIR => false
    case _ => true
  }
}

case object ValueIROnly extends Rule {
  def allows(ir: BaseIR): Boolean = ir match {
    case _: IR => true
    case _ => false
  }
}

case object EmittableValueIRs extends Rule {
  override def allows(ir: BaseIR): Boolean = ir match {
    case x: IR => Emittable(x)
    case _ => true
  }
}

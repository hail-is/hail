package is.hail.expr.ir.lowering

import is.hail.expr.ir._

trait Rule {
  def allows(ir: BaseIR): Boolean
}

case object NoMatrixIR extends Rule {
  def allows(ir: BaseIR): Boolean = !ir.isInstanceOf[MatrixIR]
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

case object StreamableIRs extends Rule {
  override def allows(ir: BaseIR): Boolean = ir match {
    case _: NA => true
    case _: In => true
    case _: ReadPartition => true
    case _: MakeStream => true
    case _: StreamRange => true
    case _: ToStream => true
    case _: Let => true
    case _: ArrayMap => true
    case _: ArrayZip => true
    case _: ArrayFilter => true
    case _: ArrayFlatMap => true
    case _: ArrayLeftJoinDistinct => true
    case _: ArrayScan => true
    case _: RunAggScan => true
    case _: ReadPartition => true
    case x if maybeAllows(x) => true
    case _ => false
  }

  def streamOnlyNode(ir: BaseIR): Boolean = ir match {
    case _: ArrayMap | _: ArrayZip | _: ArrayFilter | _: ArrayRange | _: ArrayFlatMap | _: ArrayScan |
         _: ArrayLeftJoinDistinct | _: RunAggScan | _: ArrayAggScan | _: ReadPartition | _: MakeStream | _: StreamRange => true
  }
  // matched on in both Emit and EmitStream (stream and non-stream contexts
  def maybeAllows(ir: BaseIR): Boolean = ir match {
    case _: If => true
    case _: Let => true
    case _ => false
  }

  def allowsIfChildStreamable(ir: BaseIR): Boolean = ir match {
    case ToArray(a) => allows(a)
  }
}
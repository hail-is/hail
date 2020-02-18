package is.hail.expr.ir.lowering

import is.hail.expr.ir._
import is.hail.expr.types.virtual.TStream

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

case object NoApplyIR extends Rule {
  override def allows(ir: BaseIR): Boolean = ir match {
    case x: ApplyIR => false
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
    case ArrayFold(a, _, _, _, _) => a.typ.isInstanceOf[TStream]
    case ArrayFor(a, _, _) => a.typ.isInstanceOf[TStream]
    case ArrayFold2(a, _, _, _, _) => a.typ.isInstanceOf[TStream]
    case RunAggScan(a, _, _, _, _, _) => a.typ.isInstanceOf[TStream]
    case ArrayZip(childIRs, _, _, _) => childIRs.forall(_.typ.isInstanceOf[TStream])
    case ArrayMap(a, _, _) => a.typ.isInstanceOf[TStream]
    case ArrayFilter(a, _, _) => a.typ.isInstanceOf[TStream]
    case ArrayFlatMap(a, _, b) => a.typ.isInstanceOf[TStream] && b.typ.isInstanceOf[TStream]
    case ArrayScan(a, _, _, _, _) => a.typ.isInstanceOf[TStream]
    case ArrayLeftJoinDistinct(l, r, _, _, _, _) => l.typ.isInstanceOf[TStream] && r.typ.isInstanceOf[TStream]
    case CollectDistributedArray(contextsIR, _, _, _, _) => contextsIR.typ.isInstanceOf[TStream]
    case ToDict(a) => a.typ.isInstanceOf[TStream]
    case ToSet(a) => a.typ.isInstanceOf[TStream]
    case ArraySort(a, _, _, _) => a.typ.isInstanceOf[TStream]
    case GroupByKey(collection) => collection.typ.isInstanceOf[TStream]
    case _: MakeArray => false
    case _: ArrayRange => false
    case _ => true
  }
}

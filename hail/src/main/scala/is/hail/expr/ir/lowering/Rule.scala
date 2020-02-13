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

case object ArrayIRsAreStreams extends Rule {
  override def allows(ir: BaseIR): Boolean = {
    case NA => true
    case In => true
    case ReadPartition => true
    case MakeStream => true
    case StreamRange => true
    case ToStream => true
    case Let(_, _, childIR) => {
      allows(childIR)
    }
    case ArrayMap(childIR, _, _) => {
      allows(childIR)
    }
    case ArrayZip(as, _, _,_) => {
      as.forall(allows)
    }
    case ArrayFilter(childIR, _, _,_) => {
      allows(childIR)
    }
    case ArrayFlatMap(outerIR, _, innerIR) => {
      allows(outerIR) && allows(innerIR)
    }
    case ArrayLeftJoinDistinct(leftIR, rightIR, _, _, _, _) => {
      allows(leftIR) && allows(rightIR)
    }
    case ArrayScan(childIR, _, _, _, _) => {
      allows(childIR)
    }
    case RunAggScan(array, _, _, _, _, _) => {
      allows(array)
    }
    case If(_, thenIR, elseIR) => {
      allows(thenIR) && allows(elseIR)
    }
    case ReadPartition => true
    case _ => false
  }
}
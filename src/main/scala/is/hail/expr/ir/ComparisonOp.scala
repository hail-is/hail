package is.hail.expr.ir

import is.hail.annotations.CodeOrdering
import is.hail.asm4s.Code
import is.hail.expr.types._
import is.hail.utils.lift

object ComparisonOp {

  private def checkCompatible[T](lt: Type, rt: Type): Unit =
    if (lt != rt)
      throw new RuntimeException(s"Cannot compare types $lt and $rt")

  val fromStringAndTypes: PartialFunction[(String, Type, Type), ComparisonOp] = {
    case ("==", t1, t2) =>
      checkCompatible(t1, t2)
      EQ(t1)
    case ("!=", t1, t2) =>
      checkCompatible(t1, t2)
      NEQ(t1)
    case (">=", t1, t2) =>
      checkCompatible(t1, t2)
      GTEQ(t1)
    case ("<=", t1, t2) =>
      checkCompatible(t1, t2)
      LTEQ(t1)
    case (">", t1, t2) =>
      checkCompatible(t1, t2)
      GT(t1)
    case ("<", t1, t2) =>
      checkCompatible(t1, t2)
      LT(t1)
  }
}

sealed trait ComparisonOp {
  def typ: Type
  def op: CodeOrdering.Op
  def codeOrdering(mb: EmitMethodBuilder): CodeOrdering.F[Boolean] = {
    mb.getCodeOrdering[Boolean](typ, op, missingGreatest = true)
  }
}

case class GT(typ: Type) extends ComparisonOp { val op: CodeOrdering.Op = CodeOrdering.gt }
case class GTEQ(typ: Type) extends ComparisonOp { val op: CodeOrdering.Op = CodeOrdering.gteq }
case class LTEQ(typ: Type) extends ComparisonOp { val op: CodeOrdering.Op = CodeOrdering.lteq }
case class LT(typ: Type) extends ComparisonOp { val op: CodeOrdering.Op = CodeOrdering.lt }
case class EQ(typ: Type) extends ComparisonOp { val op: CodeOrdering.Op = CodeOrdering.equiv }
case class NEQ(typ: Type) extends ComparisonOp { val op: CodeOrdering.Op = CodeOrdering.neq }
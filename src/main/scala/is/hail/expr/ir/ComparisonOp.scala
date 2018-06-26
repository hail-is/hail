package is.hail.expr.ir

import is.hail.annotations.CodeOrdering
import is.hail.asm4s.Code
import is.hail.expr.types._
import is.hail.utils.lift

object ComparisonOp {

  private def checkCompatible[T](lt: Type, rt: Type): Unit =
    if (!lt.isOfType(rt))
      throw new RuntimeException(s"Cannot compare types $lt and $rt")

  val fromStringAndTypes: PartialFunction[(String, Type, Type), ComparisonOp] = {
    case ("==" | "EQ", t1, t2) =>
      checkCompatible(t1, t2)
      EQ(t1, t2)
    case ("!=" | "NEQ", t1, t2) =>
      checkCompatible(t1, t2)
      NEQ(t1, t2)
    case (">=" | "GTEQ", t1, t2) =>
      checkCompatible(t1, t2)
      GTEQ(t1, t2)
    case ("<=" | "LTEQ", t1, t2) =>
      checkCompatible(t1, t2)
      LTEQ(t1, t2)
    case (">" | "GT", t1, t2) =>
      checkCompatible(t1, t2)
      GT(t1, t2)
    case ("<" | "LT", t1, t2) =>
      checkCompatible(t1, t2)
      LT(t1, t2)
  }
}

sealed trait ComparisonOp {
  def t1: Type
  def t2: Type
  def op: CodeOrdering.Op
  val strict: Boolean = true
  def codeOrdering(mb: EmitMethodBuilder): CodeOrdering.F[Boolean] = {
    mb.getCodeOrdering[Boolean](t1, t2, op, missingGreatest = true)
  }
}

case class GT(t1: Type, t2: Type) extends ComparisonOp { val op: CodeOrdering.Op = CodeOrdering.gt }
object GT { def apply(typ: Type): GT = GT(typ, typ) }
case class GTEQ(t1: Type, t2: Type) extends ComparisonOp { val op: CodeOrdering.Op = CodeOrdering.gteq }
object GTEQ { def apply(typ: Type): GTEQ = GTEQ(typ, typ) }
case class LTEQ(t1: Type, t2: Type) extends ComparisonOp { val op: CodeOrdering.Op = CodeOrdering.lteq }
object LTEQ { def apply(typ: Type): LTEQ = LTEQ(typ, typ) }
case class LT(t1: Type, t2: Type) extends ComparisonOp { val op: CodeOrdering.Op = CodeOrdering.lt }
object LT { def apply(typ: Type): LT = LT(typ, typ) }
case class EQ(t1: Type, t2: Type) extends ComparisonOp { val op: CodeOrdering.Op = CodeOrdering.equiv }
object EQ { def apply(typ: Type): EQ = EQ(typ, typ) }
case class NEQ(t1: Type, t2: Type) extends ComparisonOp { val op: CodeOrdering.Op = CodeOrdering.neq }
object NEQ { def apply(typ: Type): NEQ = NEQ(typ, typ) }
case class EQWithNA(t1: Type, t2: Type) extends ComparisonOp {
  val op: CodeOrdering.Op = CodeOrdering.equiv
  override val strict: Boolean = false
}
object EQWithNA { def apply(typ: Type): EQWithNA = EQWithNA(typ, typ) }
case class NEQWithNA(t1: Type, t2: Type) extends ComparisonOp {
  val op: CodeOrdering.Op = CodeOrdering.neq
  override val strict: Boolean = false
}
object NEQWithNA { def apply(typ: Type): NEQWithNA = NEQWithNA(typ, typ) }

package is.hail.expr.ir

import is.hail.annotations.CodeOrdering
import is.hail.expr.types.physical.PType
import is.hail.expr.types.virtual.Type

object ComparisonOp {

  private def checkCompatible[T](lt: Type, rt: Type): Unit =
    if (lt != rt)
      throw new RuntimeException(s"Cannot compare types $lt and $rt")

  val fromStringAndTypes: PartialFunction[(String, Type, Type), ComparisonOp[_]] = {
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
    case ("Compare", t1, t2) =>
      checkCompatible(t1, t2)
      Compare(t1, t2)
  }

  def invert[T](op: ComparisonOp[Boolean]): ComparisonOp[Boolean] = {
    assert(!op.isInstanceOf[Compare])
    op match {
      case GT(t1, t2) => LTEQ(t1, t2)
      case LT(t1, t2) => GTEQ(t1, t2)
      case GTEQ(t1, t2) => LT(t1, t2)
      case LTEQ(t1, t2) => GT(t1, t2)
      case EQ(t1, t2) => NEQ(t1, t2)
      case NEQ(t1, t2) => EQ(t1, t2)
      case EQWithNA(t1, t2) => NEQWithNA(t1, t2)
      case NEQWithNA(t1, t2) => EQWithNA(t1, t2)
    }
  }
}

sealed trait ComparisonOp[ReturnType] {
  def t1: Type
  def t2: Type
  val op: CodeOrdering.Op
  val strict: Boolean = true
  def codeOrdering(mb: EmitMethodBuilder[_], t1p: PType, t2p: PType): CodeOrdering.F[op.ReturnType] = {
    ComparisonOp.checkCompatible(t1p.virtualType, t2p.virtualType)
    mb.getCodeOrdering(t1p, t2p, op)
  }
}

case class GT(t1: Type, t2: Type) extends ComparisonOp[Boolean] { val op: CodeOrdering.Op = CodeOrdering.gt }
object GT { def apply(typ: Type): GT = GT(typ, typ) }
case class GTEQ(t1: Type, t2: Type) extends ComparisonOp[Boolean] { val op: CodeOrdering.Op = CodeOrdering.gteq }
object GTEQ { def apply(typ: Type): GTEQ = GTEQ(typ, typ) }
case class LTEQ(t1: Type, t2: Type) extends ComparisonOp[Boolean] { val op: CodeOrdering.Op = CodeOrdering.lteq }
object LTEQ { def apply(typ: Type): LTEQ = LTEQ(typ, typ) }
case class LT(t1: Type, t2: Type) extends ComparisonOp[Boolean] { val op: CodeOrdering.Op = CodeOrdering.lt }
object LT { def apply(typ: Type): LT = LT(typ, typ) }
case class EQ(t1: Type, t2: Type) extends ComparisonOp[Boolean] { val op: CodeOrdering.Op = CodeOrdering.equiv }
object EQ { def apply(typ: Type): EQ = EQ(typ, typ) }
case class NEQ(t1: Type, t2: Type) extends ComparisonOp[Boolean] { val op: CodeOrdering.Op = CodeOrdering.neq }
object NEQ { def apply(typ: Type): NEQ = NEQ(typ, typ) }
case class EQWithNA(t1: Type, t2: Type) extends ComparisonOp[Boolean] {
  val op: CodeOrdering.Op = CodeOrdering.equiv
  override val strict: Boolean = false
}
object EQWithNA { def apply(typ: Type): EQWithNA = EQWithNA(typ, typ) }
case class NEQWithNA(t1: Type, t2: Type) extends ComparisonOp[Boolean] {
  val op: CodeOrdering.Op = CodeOrdering.neq
  override val strict: Boolean = false
}
object NEQWithNA { def apply(typ: Type): NEQWithNA = NEQWithNA(typ, typ) }
case class Compare(t1: Type, t2: Type) extends ComparisonOp[Int] {
  override val strict: Boolean = false
  val op: CodeOrdering.Op = CodeOrdering.compare
}
object Compare { def apply(typ: Type): Compare = Compare(typ, typ) }

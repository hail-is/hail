package is.hail.expr.ir

import is.hail.expr.ir.orderings.CodeOrdering
import is.hail.expr.ir.orderings.CodeOrdering.F
import is.hail.types.physical.stypes.SType
import is.hail.types.physical.stypes.interfaces.SBaseStruct
import is.hail.types.virtual.Type

object ComparisonOp {

  def checkCompatible[T](lt: Type, rt: Type): Unit =
    if (lt != rt)
      throw new RuntimeException(s"Cannot compare types $lt and $rt")

  val fromString: PartialFunction[String, ComparisonOp[_]] = {
    case "==" | "EQ" =>
      EQ(null, null)
    case "!=" | "NEQ" =>
      NEQ(null, null)
    case ">=" | "GTEQ" =>
      GTEQ(null, null)
    case "<=" | "LTEQ" =>
      LTEQ(null, null)
    case ">" | "GT" =>
      GT(null, null)
    case "<" | "LT" =>
      LT(null, null)
    case "Compare" =>
      Compare(null, null)
  }

  def negate(op: ComparisonOp[Boolean]): ComparisonOp[Boolean] = {
    op match {
      case GT(t1, t2) => LTEQ(t1, t2)
      case LT(t1, t2) => GTEQ(t1, t2)
      case GTEQ(t1, t2) => LT(t1, t2)
      case LTEQ(t1, t2) => GT(t1, t2)
      case EQ(t1, t2) => NEQ(t1, t2)
      case NEQ(t1, t2) => EQ(t1, t2)
      case EQWithNA(t1, t2) => NEQWithNA(t1, t2)
      case NEQWithNA(t1, t2) => EQWithNA(t1, t2)
      case StructLT(t1, t2, sortFields) => StructGTEQ(t1, t2, sortFields)
      case StructGT(t1, t2, sortFields) => StructLTEQ(t1, t2, sortFields)
      case StructLTEQ(t1, t2, sortFields) => StructGT(t1, t2, sortFields)
      case StructGTEQ(t1, t2, sortFields) => StructLT(t1, t2, sortFields)
    }
  }

  def swap(op: ComparisonOp[Boolean]): ComparisonOp[Boolean] = {
    op match {
      case GT(t1, t2) => LT(t2, t1)
      case LT(t1, t2) => GT(t2, t1)
      case GTEQ(t1, t2) => LTEQ(t2, t1)
      case LTEQ(t1, t2) => GTEQ(t2, t1)
      case EQ(t1, t2) => EQ(t2, t1)
      case NEQ(t1, t2) => NEQ(t2, t1)
      case EQWithNA(t1, t2) => EQWithNA(t2, t1)
      case NEQWithNA(t1, t2) => NEQWithNA(t2, t1)
      case StructLT(t1, t2, sortFields) => StructGT(t2, t1, sortFields)
      case StructGT(t1, t2, sortFields) => StructLT(t2, t1, sortFields)
      case StructLTEQ(t1, t2, sortFields) => StructGTEQ(t2, t1, sortFields)
      case StructGTEQ(t1, t2, sortFields) => StructLTEQ(t2, t1, sortFields)
    }
  }
}

sealed trait ComparisonOp[ReturnType] {
  def t1: Type
  def t2: Type
  val op: CodeOrdering.Op
  val strict: Boolean = true

  def codeOrdering(ecb: EmitClassBuilder[_], t1p: SType, t2p: SType)
    : CodeOrdering.F[op.ReturnType] = {
    ComparisonOp.checkCompatible(t1p.virtualType, t2p.virtualType)
    ecb.getOrderingFunction(t1p, t2p, op).asInstanceOf[CodeOrdering.F[op.ReturnType]]
  }

  def render(): is.hail.utils.prettyPrint.Doc = Pretty.prettyClass(this)

  def copy(t1: Type, t2: Type): ComparisonOp[ReturnType]
}

case class GT(t1: Type, t2: Type) extends ComparisonOp[Boolean] {
  val op: CodeOrdering.Op = CodeOrdering.Gt()
  override def copy(t1: Type = t1, t2: Type = t2): GT = GT(t1, t2)
}

object GT { def apply(typ: Type): GT = GT(typ, typ) }

case class GTEQ(t1: Type, t2: Type) extends ComparisonOp[Boolean] {
  val op: CodeOrdering.Op = CodeOrdering.Gteq()
  override def copy(t1: Type = t1, t2: Type = t2): GTEQ = GTEQ(t1, t2)
}

object GTEQ { def apply(typ: Type): GTEQ = GTEQ(typ, typ) }

case class LTEQ(t1: Type, t2: Type) extends ComparisonOp[Boolean] {
  val op: CodeOrdering.Op = CodeOrdering.Lteq()
  override def copy(t1: Type = t1, t2: Type = t2): LTEQ = LTEQ(t1, t2)
}

object LTEQ { def apply(typ: Type): LTEQ = LTEQ(typ, typ) }

case class LT(t1: Type, t2: Type) extends ComparisonOp[Boolean] {
  val op: CodeOrdering.Op = CodeOrdering.Lt()
  override def copy(t1: Type = t1, t2: Type = t2): LT = LT(t1, t2)
}

object LT { def apply(typ: Type): LT = LT(typ, typ) }

case class EQ(t1: Type, t2: Type) extends ComparisonOp[Boolean] {
  val op: CodeOrdering.Op = CodeOrdering.Equiv()
  override def copy(t1: Type = t1, t2: Type = t2): EQ = EQ(t1, t2)
}

object EQ { def apply(typ: Type): EQ = EQ(typ, typ) }

case class NEQ(t1: Type, t2: Type) extends ComparisonOp[Boolean] {
  val op: CodeOrdering.Op = CodeOrdering.Neq()
  override def copy(t1: Type = t1, t2: Type = t2): NEQ = NEQ(t1, t2)
}

object NEQ { def apply(typ: Type): NEQ = NEQ(typ, typ) }

case class EQWithNA(t1: Type, t2: Type) extends ComparisonOp[Boolean] {
  val op: CodeOrdering.Op = CodeOrdering.Equiv()
  override val strict: Boolean = false
  override def copy(t1: Type = t1, t2: Type = t2): EQWithNA = EQWithNA(t1, t2)
}

object EQWithNA { def apply(typ: Type): EQWithNA = EQWithNA(typ, typ) }

case class NEQWithNA(t1: Type, t2: Type) extends ComparisonOp[Boolean] {
  val op: CodeOrdering.Op = CodeOrdering.Neq()
  override val strict: Boolean = false
  override def copy(t1: Type = t1, t2: Type = t2): NEQWithNA = NEQWithNA(t1, t2)
}

object NEQWithNA { def apply(typ: Type): NEQWithNA = NEQWithNA(typ, typ) }

case class Compare(t1: Type, t2: Type) extends ComparisonOp[Int] {
  override val strict: Boolean = false
  val op: CodeOrdering.Op = CodeOrdering.Compare()
  override def copy(t1: Type = t1, t2: Type = t2): Compare = Compare(t1, t2)
}

object Compare { def apply(typ: Type): Compare = Compare(typ, typ) }

trait StructComparisonOp[T] extends ComparisonOp[T] {
  val sortFields: Array[SortField]

  override def codeOrdering(ecb: EmitClassBuilder[_], t1: SType, t2: SType): F[op.ReturnType] = {
    ComparisonOp.checkCompatible(t1.virtualType, t2.virtualType)
    ecb.getStructOrderingFunction(
      t1.asInstanceOf[SBaseStruct],
      t2.asInstanceOf[SBaseStruct],
      sortFields,
      op,
    ).asInstanceOf[CodeOrdering.F[op.ReturnType]]
  }
}

case class StructCompare(t1: Type, t2: Type, sortFields: Array[SortField])
    extends StructComparisonOp[Int] {
  val op: CodeOrdering.Op = CodeOrdering.StructCompare()
  override val strict: Boolean = false
  override def copy(t1: Type = t1, t2: Type = t2): StructCompare = StructCompare(t1, t2, sortFields)
}

case class StructLT(t1: Type, t2: Type, sortFields: Array[SortField])
    extends StructComparisonOp[Boolean] {
  val op: CodeOrdering.Op = CodeOrdering.StructLt()
  override def copy(t1: Type = t1, t2: Type = t2): StructLT = StructLT(t1, t2, sortFields)
}

object StructLT {
  def apply(typ: Type, sortFields: IndexedSeq[SortField]): StructLT =
    StructLT(typ, typ, sortFields.toArray)
}

case class StructLTEQ(t1: Type, t2: Type, sortFields: Array[SortField])
    extends StructComparisonOp[Boolean] {
  val op: CodeOrdering.Op = CodeOrdering.StructLteq()
  override def copy(t1: Type = t1, t2: Type = t2): StructLTEQ = StructLTEQ(t1, t2, sortFields)
}

object StructLTEQ {
  def apply(typ: Type, sortFields: IndexedSeq[SortField]): StructLTEQ =
    StructLTEQ(typ, typ, sortFields.toArray)
}

case class StructGT(t1: Type, t2: Type, sortFields: Array[SortField])
    extends StructComparisonOp[Boolean] {
  val op: CodeOrdering.Op = CodeOrdering.StructGt()
  override def copy(t1: Type = t1, t2: Type = t2): StructGT = StructGT(t1, t2, sortFields)
}

object StructGT {
  def apply(typ: Type, sortFields: IndexedSeq[SortField]): StructGT =
    StructGT(typ, typ, sortFields.toArray)
}

case class StructGTEQ(t1: Type, t2: Type, sortFields: Array[SortField])
    extends StructComparisonOp[Boolean] {
  val op: CodeOrdering.Op = CodeOrdering.StructGteq()
  override def copy(t1: Type = t1, t2: Type = t2): StructGTEQ = StructGTEQ(t1, t2, sortFields)
}

object StructGTEQ {
  def apply(typ: Type, sortFields: IndexedSeq[SortField]): StructGTEQ =
    StructGTEQ(typ, typ, sortFields.toArray)
}
